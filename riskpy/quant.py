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
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

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


def norm_cdf(x: float) -> float:
    """Standard normal CDF, via ``math.erf`` — no SciPy, full double precision."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _check_option(kind: str) -> str:
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return kind


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float):
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
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
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive, got S={S!r} K={K!r}")
    if T < 0:
        raise ValueError(f"T must be >= 0, got {T!r}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma!r}")

    # No time left, or no uncertainty: the forward is known, so the option is
    # worth its discounted intrinsic value and nothing more.
    if T == 0 or sigma == 0:
        forward = S * math.exp(-q * T) - K * math.exp(-r * T)
        return max(forward, 0.0) if kind == "call" else max(-forward, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    discount_s = S * math.exp(-q * T)
    discount_k = K * math.exp(-r * T)

    if kind == "call":
        return discount_s * norm_cdf(d1) - discount_k * norm_cdf(d2)
    return discount_k * norm_cdf(-d2) - discount_s * norm_cdf(-d1)


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

    if T == 0 or sigma == 0:
        # Every Greek is either zero or undefined at the kink; report zeros
        # rather than raising, so a portfolio sweep containing an expiring
        # option does not blow up.
        intrinsic_delta = 0.0
        if kind == "call" and S > K:
            intrinsic_delta = 1.0
        elif kind == "put" and S < K:
            intrinsic_delta = -1.0
        return Greeks(price=price, delta=intrinsic_delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    pdf_d1 = norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t
    common_theta = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)

    if kind == "call":
        delta = disc_q * norm_cdf(d1)
        theta = common_theta - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1)
        rho = K * T * disc_r * norm_cdf(d2)
    else:
        delta = -disc_q * norm_cdf(-d1)
        theta = common_theta + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1)
        rho = -K * T * disc_r * norm_cdf(-d2)

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
    solver failed.
    """
    kind = _check_option(kind)
    if T <= 0:
        raise ValueError("implied vol is undefined at or after expiry")

    lower_bound = black_scholes(S, K, T, r, 0.0, q, kind)
    if price < lower_bound - 1e-12:
        raise ValueError(
            f"price {price!r} is below intrinsic ({lower_bound:.6f}) — no volatility "
            f"can produce it, so the quote violates no-arbitrage."
        )

    low, high = 1e-9, 5.0
    # Widen the upper bracket if a genuinely extreme quote needs it.
    for _ in range(10):
        if black_scholes(S, K, T, r, high, q, kind) >= price:
            break
        high *= 2.0
    else:
        raise ValueError(
            f"price {price!r} exceeds what even {high:.0f}00% volatility produces — "
            f"check the inputs."
        )

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        value = black_scholes(S, K, T, r, mid, q, kind)
        if abs(value - price) < tolerance:
            return mid
        if value < price:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _numpy():
    from .mc import _numpy as _np

    return _np()


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
    if steps < 1 or trials < 1:
        raise ValueError("steps and trials must both be >= 1")

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


def historical_var(returns: Sequence[float], level: float = 0.99) -> float:
    """VaR from the empirical distribution of returns, as a positive loss.

    No distributional assumption, which is the point — and the limitation, since
    it can never report a loss worse than the worst one in your sample.
    """
    np = _numpy()
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    data = np.asarray(list(returns), dtype=float)
    if data.size == 0:
        raise ValueError("returns is empty")
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
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")

    if mean is None or sd is None:
        data = np.asarray(list(returns), dtype=float)
        if data.size < 2:
            raise ValueError("need at least two returns, or explicit mean and sd")
        mean = float(data.mean()) if mean is None else mean
        sd = float(data.std(ddof=1)) if sd is None else sd

    z = _normal_quantile(1.0 - level)
    return float(-(mean + z * sd))


def expected_shortfall(returns: Sequence[float], level: float = 0.99) -> float:
    """Mean loss in the worst ``1 - level`` of outcomes, as a positive number."""
    np = _numpy()
    data = np.asarray(list(returns), dtype=float)
    cutoff = np.percentile(data, (1.0 - level) * 100.0)
    tail = data[data <= cutoff]
    if tail.size == 0:  # pragma: no cover
        return float(-cutoff)
    return float(-tail.mean())


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF — :func:`riskpy._special.norm_ppf`.

    Kept under its old name because other modules import it. It used to be
    Acklam's rational approximation on its own (relative error 1e-9); it is now
    that approximation plus one Halley refinement, exact to double precision.
    """
    return _special.norm_ppf(p)


# ---------------------------------------------------------------------------
# Transform pricing
# ---------------------------------------------------------------------------


def _heston_cf(u: complex, T: float, r: float, q: float, v0: float,
               kappa: float, theta: float, xi: float, rho: float) -> complex:
    """Characteristic function of the log-price under Heston.

    The "little Heston trap" formulation: written with ``g = C/D`` rather than
    ``D/C`` so the complex logarithm stays on its principal branch for long
    maturities. The textbook form is algebraically identical and numerically
    wrong past a couple of years.
    """
    i = 1j
    d = cmath.sqrt((rho * xi * i * u - kappa) ** 2 + xi * xi * (i * u + u * u))
    g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)

    exp_dt = cmath.exp(-d * T)
    C = (r - q) * i * u * T + (kappa * theta / (xi * xi)) * (
        (kappa - rho * xi * i * u - d) * T - 2.0 * cmath.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = ((kappa - rho * xi * i * u - d) / (xi * xi)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
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
    not enforced, because real calibrations violate it routinely.

    Integration is Simpson's rule on ``[0, upper]``. Raise ``nodes`` if you are
    pricing very short maturities, where the integrand oscillates fastest.
    """
    np = _numpy()
    kind = _check_option(kind)

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
    # the strike enters as log-moneyness, not as log(K).
    log_k = math.log(K / S)

    # Gil-Pelaez inversion: two probabilities, P1 under the share measure and
    # P2 under the risk-neutral one.
    def integrand(shift: int):
        out = np.empty(u.shape, dtype=float)
        for idx, ui in enumerate(u):
            if shift == 1:
                numerator = _heston_cf(ui - 1j, T, r, q, v0, kappa, theta, xi, rho)
                denominator = _heston_cf(-1j, T, r, q, v0, kappa, theta, xi, rho)
                phi = numerator / denominator
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
