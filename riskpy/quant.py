"""Quantitative finance — analytic pricing, Greeks, paths and portfolio risk.

Three layers, deliberately separated by what they cost you to install:

* **Closed form** (:func:`black_scholes`, :func:`greeks`, :func:`implied_vol`)
  is pure Python and needs nothing at all, so it works in the same lean install
  as the C++ core.
* **Path simulation** (:func:`gbm_paths`, :func:`merton_jump_paths`) and
  **portfolio risk** (:func:`historical_var`, :func:`parametric_var`) need
  NumPy — ``pip install open-riskpy[sim]``.
* **Transform pricing** (:func:`heston_price`) needs NumPy too. It is the
  answer to "the formula is too big": when a model has no closed-form price but
  a known characteristic function, you integrate the transform instead of
  simulating, and get the whole strike surface in one pass.

Conventions throughout: ``T`` is in years, rates and volatilities are annualised
decimals (``0.05``, not ``5``), and ``q`` is the continuous dividend yield.
"""

from __future__ import annotations

import cmath
import math

from . import _special
from ._validate import _check_finite, _check_non_negative, _check_number, _check_positive, _check_seed, _check_trials
from .errors import ConvergenceError, InputError
from .mc import _numpy
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "black_scholes",
    "greeks",
    "Greeks",
    "implied_vol",
    "gbm_paths",
    "merton_jump_paths",
    "historical_var",
    "parametric_var",
    "expected_shortfall",
    "heston_price",
    "norm_cdf",
    "norm_pdf",
]

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_EPS = 2.220446049250313e-16
_PLAIN = frozenset({float, int})


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT_2)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def norm_cdf(x: float) -> float:
    """Standard normal CDF, via ``math.erfc`` — no SciPy, full relative precision
    in the lower tail (``erf`` rounds to -1 below x ≈ -8, which priced a far
    out-of-the-money put at exactly 0 and a 3×-strike call 8% wrong).

    ``x`` must be a number (``±inf`` give 1 and 0, NaN gives NaN); ``None``, a
    string or a bool raises RP101 rather than a bare ``TypeError``.
    """
    return _norm_cdf(_check_number("x", x))


def norm_pdf(x: float) -> float:
    """Standard normal density; ``x`` is checked as in :func:`norm_cdf`."""
    return _norm_pdf(_check_number("x", x))


def _check_option(kind: str) -> str:
    if not isinstance(kind, str):
        raise InputError(f"kind must be 'call' or 'put', got {kind!r}", code="RP108", param="kind",
                         hint='kind is "call" or "put" (case-insensitive).')
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise InputError(f"kind must be 'call' or 'put', got {kind!r}", code="RP108", param="kind",
                         hint='kind is "call" or "put" (case-insensitive).')
    return kind


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float):
    vol_sqrt_t = sigma * math.sqrt(T)
    # ln S − ln K, not ln(S/K): the quotient of two valid prices (1e-200 and 1e200)
    # underflows to 0 and its logarithm is a bare "math domain error".
    d1 = (math.log(S) - math.log(K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


# ---------------------------------------------------------------------------
# Closed form
# ---------------------------------------------------------------------------


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
) -> float:
    """Black–Scholes–Merton price of a European option.

        >>> round(black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2), 4)
        10.4506

    The degenerate cases are handled rather than allowed to divide by zero:
    at expiry, or at zero volatility, the price is the discounted intrinsic
    value.
    """
    kind = _check_option(kind)
    try:
        # Exact float/int types only: bool is an int subclass that math.isfinite accepts, and
        # anything else (NumPy scalars, strings, None) takes the checked path below.
        finite = (type(S) in _PLAIN and type(K) in _PLAIN and type(T) in _PLAIN and type(r) in _PLAIN
                  and type(sigma) in _PLAIN and type(q) in _PLAIN
                  and math.isfinite(S) and math.isfinite(K) and math.isfinite(T) and math.isfinite(r)
                  and math.isfinite(sigma) and math.isfinite(q))
    except (TypeError, OverflowError):
        finite = False
    if not finite:
        # The slow path names the bad argument; the fast one above keeps a pricing loop at
        # the cost of a few type and isfinite checks.
        S, K, T, r, sigma, q = (_check_finite(name, value) for name, value in
                                (("S", S), ("K", K), ("T", T), ("r", r), ("sigma", sigma), ("q", q)))
    if S <= 0 or K <= 0:
        raise InputError(f"S and K must be positive, got S={S!r} K={K!r}", code="RP102",
                         param="S" if not S > 0 else "K",
                         hint="Spot S and strike K are prices in money and must be > 0.")
    if T < 0:
        raise InputError(f"T must be >= 0, got {T!r}", code="RP106", param="T",
                         hint="T is the time to expiry in years (3 months is 0.25); 0 means at expiry.")
    if sigma < 0:
        raise InputError(f"sigma must be >= 0, got {sigma!r}", code="RP105", param="sigma",
                         hint="sigma is the annual volatility as a decimal (0.2 means 20%) and cannot be negative.")

    # No time left, or no uncertainty: the forward is known, so the option is
    # worth its discounted intrinsic value and nothing more.
    if T == 0 or sigma == 0:
        forward = S * math.exp(-q * T) - K * math.exp(-r * T)
        return max(forward, 0.0) if kind == "call" else max(-forward, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    discount_s = S * math.exp(-q * T)
    discount_k = K * math.exp(-r * T)

    # Far out of the money the two terms are nearly equal and rounding can
    # leave a difference of -1e-323; an option is never worth less than 0.
    if kind == "call":
        return max(discount_s * _norm_cdf(d1) - discount_k * _norm_cdf(d2), 0.0)
    return max(discount_k * _norm_cdf(-d2) - discount_s * _norm_cdf(-d1), 0.0)


@dataclass(frozen=True)
class Greeks:
    """Sensitivities, in the units people actually quote them in.

    ``vega`` is per **1 percentage point** of volatility and ``theta`` is per
    **calendar day**, because that is how they are read on a desk. ``delta``,
    ``gamma`` and ``rho`` are per unit, with ``rho`` per 1 percentage point of
    rate. Multiply vega and rho by 100, and theta by 365, to get the raw
    per-unit derivatives.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def __str__(self) -> str:
        return (
            f"price {self.price:,.4f}   delta {self.delta:+.4f}   gamma {self.gamma:+.6f}\n"
            f"vega  {self.vega:+.4f}/pt  theta {self.theta:+.4f}/day  rho {self.rho:+.4f}/pt"
        )


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
) -> Greeks:
    """Price and the five first-order Greeks for a European option."""
    kind = _check_option(kind)
    price = black_scholes(S, K, T, r, sigma, q, kind)
    # black_scholes has validated every input; work in floats from here on.
    S, K, T, r, sigma, q = float(S), float(K), float(T), float(r), float(sigma), float(q)

    if T == 0:
        # Every Greek is either zero or undefined at the kink; report zeros
        # rather than raising, so a portfolio sweep containing an expiring
        # option does not blow up.
        intrinsic_delta = 0.0
        if kind == "call" and S > K:
            intrinsic_delta = 1.0
        elif kind == "put" and S < K:
            intrinsic_delta = -1.0
        return Greeks(price=price, delta=intrinsic_delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    if sigma == 0:
        # Zero volatility with time left: the payoff is the deterministic
        # forward S·e^(−qT) − K·e^(−rT), so moneyness is decided by the
        # forward, not by S against K, and theta and rho are the derivatives of
        # that discounted intrinsic value — not zero.
        disc_s = S * math.exp(-q * T)
        disc_k = K * math.exp(-r * T)
        sign = 1.0 if kind == "call" else -1.0
        if sign * (disc_s - disc_k) <= 0.0:
            return Greeks(price=price, delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
        return Greeks(
            price=price,
            delta=sign * math.exp(-q * T),
            gamma=0.0,
            vega=0.0,
            theta=sign * (q * disc_s - r * disc_k) / 365.0,
            rho=sign * K * T * math.exp(-r * T) / 100.0,
        )

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    pdf_d1 = _norm_pdf(d1)

    # Divided by S last: a spot of 5e-324 makes S·σ·√T underflow to 0, and the density
    # (which is 0 there too) would then raise ZeroDivisionError instead of giving gamma 0.
    gamma = disc_q * pdf_d1 / (sigma * sqrt_t) / S
    vega = S * disc_q * pdf_d1 * sqrt_t
    common_theta = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)

    if kind == "call":
        delta = disc_q * _norm_cdf(d1)
        theta = common_theta - r * K * disc_r * _norm_cdf(d2) + q * S * disc_q * _norm_cdf(d1)
        rho = K * T * disc_r * _norm_cdf(d2)
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta = common_theta + r * K * disc_r * _norm_cdf(-d2) - q * S * disc_q * _norm_cdf(-d1)
        rho = -K * T * disc_r * _norm_cdf(-d2)

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega / 100.0,
        theta=theta / 365.0,
        rho=rho / 100.0,
    )


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    kind: str = "call",
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> float:
    """Back out the volatility that reproduces ``price``.

    Bisection on a bracketed interval rather than Newton. Newton is faster when
    it works, but vega collapses for deep in- or out-of-the-money options and it
    then diverges — bisection cannot, and 200 iterations of it is still
    instant.

    Raises ``ValueError`` if the price is outside the no-arbitrage bounds, which
    is the useful answer: it means the quote is stale or wrong, not that the
    solver failed. It also raises when the price is so close to a bound that
    double precision cannot tell volatilities apart (a 2000%-vol call is worth
    S to sixteen digits): any answer then would be arbitrary.

    ``tolerance`` is on the price, relative to it once the price is below 1, so
    a far out-of-the-money quote of 1e-12 is matched to its own digits rather
    than "converged" at the first midpoint.
    """
    kind = _check_option(kind)
    price = _check_finite("price", price)
    # Checked before T is compared with 0, so None or a string is RP101, not a TypeError.
    S, K, T, r, q = (_check_finite(name, value) for name, value in
                     (("S", S), ("K", K), ("T", T), ("r", r), ("q", q)))
    tolerance = _check_positive("tolerance", tolerance)
    max_iterations = _check_trials("max_iterations", max_iterations)
    if price < 0:
        raise InputError(f"price must be >= 0, got {price!r}", code="RP102", param="price",
                         hint="An option price is a non-negative amount of money per unit.")
    if T <= 0:
        raise InputError(f"implied vol is undefined at or after expiry: T must be > 0, got {T!r}", code="RP106",
                         param="T", hint="At expiry an option is worth its intrinsic value whatever the volatility, "
                         "so there is nothing to invert. Pass the time to expiry in years (> 0).")

    lower_bound = black_scholes(S, K, T, r, 0.0, q, kind)
    if price < lower_bound - 1e-12:
        raise ConvergenceError(
            f"price {price!r} is below intrinsic ({lower_bound:.6f}) — no volatility "
            f"can produce it, so the quote violates no-arbitrage.", code="RP301", param="price",
            hint=f"The {kind}'s floor here is {lower_bound:.6f} (its discounted intrinsic value). Check the price, "
                 f"and that kind={kind!r} matches the quote.",
        )

    if price - lower_bound <= 8.0 * _EPS * max(abs(price), abs(lower_bound), 1e-300):
        raise ConvergenceError(
            f"price {price!r} equals the option's intrinsic floor ({lower_bound:.6g}) in double precision, so it "
            f"has no measurable time value and does not determine a volatility.", code="RP301", param="price",
            hint="Implied vol is only defined where the option has time value; for a deep in-the-money quote, "
                 "invert the other side via put-call parity.",
        )

    low, high = 1e-9, 5.0
    # Widen the upper bracket if a genuinely extreme quote needs it.
    for _ in range(10):
        if black_scholes(S, K, T, r, high, q, kind) >= price:
            break
        high *= 2.0
    else:
        ceiling = S * math.exp(-q * T) if kind == "call" else K * math.exp(-r * T)
        raise ConvergenceError(
            f"price {price!r} exceeds what even {high:.0f}00% volatility produces — "
            f"check the inputs.", code="RP302", param="price",
            hint=f"No volatility gives more than {ceiling:.6f} for this {kind} "
                 f"({'S·e^(−qT)' if kind == 'call' else 'K·e^(−rT)'}). Check the price is per unit, and S and K.",
        )

    price_tolerance = tolerance * min(1.0, price) if price > 0 else tolerance
    mid = 0.5 * (low + high)
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        value = black_scholes(S, K, T, r, mid, q, kind)
        # Stop when the price matches and the vol bracket is tight too: deep in
        # the money vega is tiny, and a price match alone left 4e-5 vol error.
        if (abs(value - price) < price_tolerance and high - low < 1e-10 * high) or high - low < 1e-15 * high:
            break
        if value < price:
            low = mid
        else:
            high = mid

    # A volatility is only an answer if the price pins it down: if moving it by
    # 1% leaves the price unchanged in double precision, every vol in a wide
    # range is equally "right" and bisection just returned one of them.
    bumped_down = black_scholes(S, K, T, r, mid * 0.99, q, kind)
    bumped_up = black_scholes(S, K, T, r, mid * 1.01, q, kind)
    if bumped_up - bumped_down <= 8.0 * _EPS * max(abs(price), 1e-300):
        at_floor = price - lower_bound <= 8.0 * _EPS * max(abs(price), 1e-300)
        raise ConvergenceError(
            f"price {price!r} is indistinguishable from the {'intrinsic floor' if at_floor else 'no-arbitrage ceiling'} "
            f"in double precision, so it does not determine a volatility (anything near {mid:.4g} reproduces it).",
            code="RP301" if at_floor else "RP302", param="price",
            hint="Implied vol is only defined where the option has measurable time value; use a quote with more "
                 "time value, or the option's other side via put-call parity.",
        )
    return mid


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int = 252,
    trials: int = 10_000,
    seed: Optional[int] = None,
):
    """Geometric Brownian Motion paths, shape ``(trials, steps + 1)``.

    Uses the exact log-Euler solution, not an Euler approximation of the SDE —
    GBM has a closed-form transition density, so there is no discretisation
    error to accept here and no reason to accept one.

    Feed the output straight to :func:`riskpy.viz.fan`.
    """
    np = _numpy()
    steps = _check_trials("steps", steps)
    trials = _check_trials("trials", trials)
    seed = _check_seed(seed)
    S0 = _check_positive("S0", S0)
    mu = _check_finite("mu", mu)
    sigma = _check_non_negative("sigma", sigma)
    T = _check_non_negative("T", T)

    dt = T / steps
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((trials, steps))
    increments = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * shocks

    log_paths = np.concatenate(
        [np.zeros((trials, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


def merton_jump_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    jump_intensity: float,
    jump_mean: float,
    jump_sd: float,
    steps: int = 252,
    trials: int = 10_000,
    seed: Optional[int] = None,
):
    """Merton jump-diffusion paths, shape ``(trials, steps + 1)``.

    GBM plus Poisson jumps whose log-sizes are normal. The drift carries the
    usual compensator so the process keeps expected return ``mu`` — without it
    the jumps quietly add drift and the model prices nothing sensibly.
    """
    np = _numpy()
    steps = _check_trials("steps", steps)
    trials = _check_trials("trials", trials)
    seed = _check_seed(seed)
    S0 = _check_positive("S0", S0)
    mu = _check_finite("mu", mu)
    sigma = _check_non_negative("sigma", sigma)
    T = _check_non_negative("T", T)
    jump_intensity = _check_non_negative("jump_intensity", jump_intensity)
    jump_mean = _check_finite("jump_mean", jump_mean)
    jump_sd = _check_non_negative("jump_sd", jump_sd)
    dt = T / steps
    rng = np.random.default_rng(seed)

    compensator = jump_intensity * (math.exp(jump_mean + 0.5 * jump_sd * jump_sd) - 1.0)
    diffusion = (mu - compensator - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * rng.standard_normal((trials, steps))

    counts = rng.poisson(jump_intensity * dt, (trials, steps))
    # Sum of N normal jumps is normal with mean N*m and variance N*s² — one draw
    # per step instead of a loop over individual jumps.
    jumps = counts * jump_mean + np.sqrt(counts) * jump_sd * rng.standard_normal((trials, steps))

    log_paths = np.concatenate(
        [np.zeros((trials, 1)), np.cumsum(diffusion + jumps, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# Portfolio risk
# ---------------------------------------------------------------------------


def _check_level(level: float) -> float:
    level = _check_finite("level", level)
    if not (0.0 < level < 1.0):
        raise InputError(f"level must be in (0, 1), got {level!r}", code="RP110", param="level")
    return level


def _returns_array(np, returns, name: str = "returns"):
    """``returns`` as a 1-D float array: one return per period, numbers only.

    A 2-D array (several assets, or a column vector of shape (n, 1) from pandas) used
    to be flattened into one pooled sample without a word, which is a different
    question from the VaR of any one series or of the portfolio.
    """
    if isinstance(returns, (str, bytes)):
        raise InputError(f"{name} must be a sequence of numbers, got {returns!r}", code="RP101", param=name)
    try:
        data = np.asarray(returns if hasattr(returns, "__array__") else list(returns))
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a flat sequence of numbers, got {type(returns).__name__}",
                         code="RP101", param=name) from None
    if data.ndim != 1:
        raise InputError(
            f"{name} must be one-dimensional (one return per period), got shape {data.shape}", code="RP100",
            param=name, hint="Pass one series. For several assets, form the portfolio's return series first "
                             "(weights @ returns), or call once per column.")
    if data.dtype.kind not in "iuf":
        try:
            if data.dtype.kind != "O":  # text, booleans, complex, dates
                raise TypeError
            data = data.astype(float)   # Decimal and friends convert; None does not
        except (TypeError, ValueError):
            raise InputError(f"{name} must contain numbers only, got {data.dtype} values", code="RP101",
                             param=name, hint="Strings, booleans and None are not returns.") from None
    return data.astype(float, copy=False)


def historical_var(returns: Sequence[float], level: float = 0.99) -> float:
    """VaR from the empirical distribution of returns, as a positive loss.

    No distributional assumption, which is the point — and the limitation, since
    it can never report a loss worse than the worst one in your sample.
    ``returns`` is one series (a 1-D sequence); a 2-D array raises RP100 rather
    than being flattened.
    """
    np = _numpy()
    level = _check_level(level)
    data = _returns_array(np, returns)
    if data.size == 0:
        raise InputError("returns is empty", code="RP111", param="returns")
    if not np.all(np.isfinite(data)):
        raise InputError("returns contains NaN or infinity", code="RP101", param="returns")
    return float(-np.percentile(data, (1.0 - level) * 100.0))


def parametric_var(
    returns: Sequence[float] = (),
    level: float = 0.99,
    mean: Optional[float] = None,
    sd: Optional[float] = None,
) -> float:
    """Normal (variance–covariance) VaR, as a positive loss.

    Pass ``returns`` to estimate the moments, or ``mean`` and ``sd`` directly.
    Understates tail risk whenever returns are fat-tailed, which is nearly
    always — compare it against :func:`historical_var` rather than trusting it
    alone.
    """
    np = _numpy()
    level = _check_level(level)

    if mean is None or sd is None:
        data = _returns_array(np, returns)
        if data.size < 2:
            raise InputError("need at least two returns, or explicit mean and sd", code="RP111", param="returns")
        mean = float(data.mean()) if mean is None else mean
        sd = float(data.std(ddof=1)) if sd is None else sd
    mean = _check_finite("mean", mean)
    sd = _check_non_negative("sd", sd)

    z = _special.norm_ppf(1.0 - level)
    return float(-(mean + z * sd))


def expected_shortfall(returns: Sequence[float], level: float = 0.99) -> float:
    """Mean loss in the worst ``1 - level`` of outcomes, as a positive number."""
    np = _numpy()
    level = _check_level(level)
    data = _returns_array(np, returns)
    if data.size == 0:
        raise InputError("returns is empty", code="RP111", param="returns")
    if not np.all(np.isfinite(data)):
        raise InputError("returns contains NaN or infinity", code="RP101", param="returns")
    cutoff = np.percentile(data, (1.0 - level) * 100.0)
    tail = data[data <= cutoff]
    if tail.size == 0:  # pragma: no cover
        return float(-cutoff)
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# Transform pricing
# ---------------------------------------------------------------------------


def _cexpm1(z: complex) -> complex:
    """``e^z − 1`` for complex ``z`` without cancellation near 0 (``cmath`` has no ``expm1``)."""
    x, y = z.real, z.imag
    half_sin = math.sin(0.5 * y)
    return complex(math.expm1(x) * math.cos(y) - 2.0 * half_sin * half_sin, math.exp(x) * math.sin(y))


def _clog1p(z: complex) -> complex:
    """Principal ``ln(1 + z)`` for complex ``z``, accurate relative to ``z`` near 0."""
    x, y = z.real, z.imag
    if abs(x) < 0.5 and abs(y) < 0.5:
        # ln|1 + z| = ½·ln(1 + 2x + x² + y²): log1p keeps the digits ln(1 + tiny) loses.
        return complex(0.5 * math.log1p(x * (2.0 + x) + y * y), math.atan2(y, 1.0 + x))
    return cmath.log(1.0 + z)


def _heston_cf(u: complex, T: float, r: float, q: float, v0: float,
               kappa: float, theta: float, xi: float, rho: float) -> complex:
    """Characteristic function of the log-price under Heston.

    The "little Heston trap" formulation: written with ``g = C/D`` rather than
    ``D/C`` so the complex logarithm stays on its principal branch for long
    maturities. The textbook form is algebraically identical and numerically
    wrong past a couple of years.

    With ``b = κ − ρξiu`` and ``d = √(b² + ξ²(iu + u²))``, the two quantities
    ``b ± d`` multiply to ``−ξ²(iu + u²)``. The larger is formed by addition and
    the smaller from that product, never by subtracting nearly equal numbers: as
    ``ξ → 0``, ``b − d`` is O(ξ²) and was divided by ``ξ²``, so its rounding error
    came out O(1) and a vol-of-vol of 1e-9 priced an at-the-money call at 38.6
    instead of Black–Scholes' 9.41. For the same reason the logarithm in ``C`` is
    taken as ``log1p`` of a quantity proportional to ``ξ²`` and divided by ``ξ²`` in
    closed form, so the limit ``ξ → 0`` is Black–Scholes to rounding.
    """
    i = 1j
    a = i * u + u * u
    if a == 0:
        # u = 0 or u = −i: the variance terms vanish identically.
        return cmath.exp((r - q) * i * u * T)
    xi2 = xi * xi
    b = kappa - rho * xi * i * u
    d = cmath.sqrt(b * b + xi2 * a)
    s, t = b + d, b - d          # s·t = −ξ²·a
    if abs(s) >= abs(t):
        m = -a / s               # (b − d)/ξ², without dividing by ξ²
        g = xi2 * m / s          # (b − d)/(b + d)
        g_over_xi2 = m / s
    else:                        # b + d is the smaller: the share-measure (P1) integrand
        m = t / xi2              # when κ − ρξ is small; ξ is not small there
        s = -xi2 * a / t
        g = t / s
        g_over_xi2 = g / xi2

    exp_dt = cmath.exp(-d * T)
    one_minus_exp = -_cexpm1(-d * T)
    D = m * one_minus_exp / (1.0 - g * exp_dt)
    # ln[(1 − g·e^{−dT}) / (1 − g)] = ln(1 + z) with z = g·w, w = (1 − e^{−dT}) / (1 − g),
    # and 1 − g = 2d/s exactly. Divided by ξ²: (g/ξ²)·w·ln(1 + z)/z.
    w = one_minus_exp * s / (2.0 * d)
    z = g * w
    log_ratio = _clog1p(z) / z if z != 0 else 1.0
    log_over_xi2 = g_over_xi2 * w * log_ratio
    C = (r - q) * i * u * T + kappa * theta * (m * T - 2.0 * log_over_xi2)
    return cmath.exp(C + D * v0)


def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    q: float = 0.0,
    kind: str = "call",
    upper: float = 200.0,
    nodes: int = 4096,
) -> float:
    """European option under Heston stochastic volatility, by Fourier inversion.

    This is the concrete answer to "the formula is too big". Heston has no
    closed-form price, but it does have a closed-form characteristic function —
    so instead of simulating a million paths you integrate the transform once,
    which is both faster and exact to the quadrature error.

    Parameters follow the usual convention: ``v0`` initial variance, ``kappa``
    mean-reversion speed, ``theta`` long-run variance, ``xi`` vol-of-vol,
    ``rho`` spot/vol correlation (negative for equities). The Feller condition
    ``2·kappa·theta > xi²`` keeps variance away from zero; it is warned about,
    not enforced, because real calibrations violate it routinely. ``kappa`` must
    exceed ``rho·xi`` (RP113 otherwise): only a positive correlation with a large
    vol-of-vol breaks that, and the transform does not converge there.

    As ``xi → 0`` with ``v0 = theta`` the price tends to Black–Scholes at
    ``sigma = sqrt(theta)``, and it gets there: the characteristic function is
    written so that nothing of order ``xi²`` is formed by cancellation, so a
    vol-of-vol of 1e-12 prices like 1e-4 does.

    Integration is Simpson's rule on ``[0, upper]``. Raise ``nodes`` if you are
    pricing very short maturities, where the integrand oscillates fastest.
    """
    np = _numpy()
    kind = _check_option(kind)
    S, K = _check_positive("S", S), _check_positive("K", K)
    T = _check_non_negative("T", T)
    r, q = _check_finite("r", r), _check_finite("q", q)
    v0 = _check_non_negative("v0", v0)
    kappa = _check_positive("kappa", kappa)
    theta = _check_non_negative("theta", theta)
    xi = _check_positive("xi", xi)
    rho = _check_finite("rho", rho)
    if not -1.0 <= rho <= 1.0:
        raise InputError(f"rho must be in [-1, 1], got {rho!r}", code="RP102", param="rho",
                         hint="rho is the correlation between spot and variance shocks (about -0.7 for equities).")
    upper = _check_positive("upper", upper)
    nodes = _check_trials("nodes", nodes, minimum=2)
    if T == 0:
        # At expiry the option is its intrinsic value; the transform's
        # quadrature error would otherwise leave a small negative price.
        return float(max(S - K, 0.0) if kind == "call" else max(K - S, 0.0))

    if kappa <= rho * xi:
        # Under the share measure (the P1 integral) variance mean-reverts at κ − ρξ. At or
        # below zero it drifts away instead, the log-price spreads over a range too wide for
        # a fixed Simpson grid, and the integral returns nonsense (call prices thousands of
        # times the spot at T = 20). 0.7.0 divided by zero here; refuse it by name.
        raise InputError(
            f"heston_price needs kappa > rho·xi, got kappa = {kappa!r} and rho·xi = {rho * xi!r}: below that the "
            f"variance has no mean reversion under the share measure and the transform integral does not "
            f"converge reliably", code="RP113", param="kappa",
            hint="This needs a positive rho with a large vol-of-vol. Raise kappa, or lower rho or xi; for these "
                 "parameters, price by simulation instead.",
        )

    if 2.0 * kappa * theta <= xi * xi:
        import warnings

        warnings.warn(
            f"Feller condition violated (2·kappa·theta = {2 * kappa * theta:.4f} "
            f"<= xi² = {xi * xi:.4f}). The price is still computable, but variance "
            f"can reach zero and a simulation of the same parameters will behave "
            f"badly.",
            RuntimeWarning,
            stacklevel=2,
        )

    if nodes % 2 == 1:
        nodes += 1  # Simpson needs an even number of intervals

    u = np.linspace(1e-10, upper, nodes + 1)
    # The characteristic function above is of the log *return* ln(S_T/S), so
    # the strike enters as log-moneyness, not as log(K). (A difference of logs:
    # the quotient of two valid prices can underflow.)
    log_k = math.log(K) - math.log(S)
    # φ(−i) = E[S_T/S] = e^{(r−q)T} exactly: the discounted price is a martingale.
    # Evaluating the transform there divided 0 by 0 whenever κ < ρξ.
    forward_growth = math.exp((r - q) * T)

    # Gil-Pelaez inversion: two probabilities, P1 under the share measure and
    # P2 under the risk-neutral one.
    def integrand(shift: int):
        out = np.empty(u.shape, dtype=float)
        for idx, ui in enumerate(u):
            if shift == 1:
                phi = _heston_cf(ui - 1j, T, r, q, v0, kappa, theta, xi, rho) / forward_growth
            else:
                phi = _heston_cf(ui, T, r, q, v0, kappa, theta, xi, rho)
            out[idx] = (cmath.exp(-1j * ui * log_k) * phi / (1j * ui)).real
        return out

    def simpson(y):
        h = (u[-1] - u[0]) / nodes
        return (h / 3.0) * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum())

    p1 = 0.5 + simpson(integrand(1)) / math.pi
    p2 = 0.5 + simpson(integrand(2)) / math.pi

    call = S * math.exp(-q * T) * p1 - K * math.exp(-r * T) * p2
    if kind == "call":
        return float(call)
    # Put-call parity, rather than a second integration.
    return float(call - S * math.exp(-q * T) + K * math.exp(-r * T))
