"""Generic Monte Carlo — any formula, any distributions, any dependence.

The C++ ``MonteCarloSimulator`` covers five specific actuarial scenarios very
fast. This module covers the other case: you have a formula that is too awkward
to solve analytically, so you feed it random inputs a hundred thousand times and
look at the distribution that falls out.

    from riskpy.mc import Model, LogNormal, Normal, Poisson

    model = Model(
        claims   = Poisson(mean=120),
        severity = LogNormal(mu=8.5, sigma=1.2),
        inflation= Normal(mean=0.04, sd=0.01),
    )
    model.correlate("claims", "inflation", 0.3)     # optional dependence

    @model.formula
    def annual_loss(claims, severity, inflation):
        return claims * severity * (1 + inflation)

    result = model.run(100_000, seed=42, sampling="lhs")
    print(result.summary())
    result.plot()

Everything is vectorised: your formula is called **once** with arrays, not once
per trial, so a hundred thousand trials is one NumPy expression. If your formula
cannot be written that way, pass ``vectorised=False`` and it will be looped
instead — correct, just slower.

Every distribution also carries its analytic layer — ``pdf``, ``cdf``, ``ppf``
and ``moments()`` — so the same object that feeds a simulation can be fitted,
discretised for a Panjer recursion, or checked against the sample it produced.

NumPy is required here and is an optional extra for the package as a whole
(``pip install open-riskpy[sim]``). The C++ core keeps its zero-dependency
install; this module is opt-in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from . import _special

__all__ = [
    "Model",
    "Result",
    "simulate",
    "Distribution",
    "Moments",
    "Constant",
    "Uniform",
    "Normal",
    "LogNormal",
    "Triangular",
    "PERT",
    "Exponential",
    "Gamma",
    "Beta",
    "Pareto",
    "Weibull",
    "StudentT",
    "Poisson",
    "NegativeBinomial",
    "Bernoulli",
    "Binomial",
    "Empirical",
    "Categorical",
    "Mixture",
    "Truncated",
    "iman_conover",
    "correlation_matrix",
]


def _numpy():
    """Import NumPy on use, with an error that says how to fix it."""
    try:
        import numpy  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "riskpy.mc needs NumPy, which is not installed. The C++ core is "
            "deliberately dependency-free, so the simulation layer ships as an "
            "extra:\n"
            "    pip install open-riskpy[sim]\n"
            "or just:\n"
            "    pip install numpy"
        ) from exc
    return numpy


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Moments:
    """Mean and variance of a distribution, with the standard deviation and
    coefficient of variation derived. ``inf`` where the moment does not exist."""

    mean: float
    variance: float

    @property
    def sd(self) -> float:
        return math.sqrt(self.variance) if math.isfinite(self.variance) else math.inf

    @property
    def cv(self) -> float:
        if self.mean == 0.0 or not math.isfinite(self.mean):
            return math.inf
        return self.sd / abs(self.mean)


class Distribution:
    """Base class. A distribution has to know how to draw ``n`` samples.

    Most also carry an analytic layer — :meth:`pdf`, :meth:`cdf`, :meth:`ppf`
    and :meth:`moments` — which is what makes Latin hypercube sampling,
    correlated inputs, distribution fitting and Panjer discretisation possible.
    Where a closed form is not available the base class falls back to a
    safeguarded Newton inversion of the CDF, and says so.

    Parameters are validated in ``__init__`` rather than at sample time. A
    negative standard deviation is a typo, and finding out about it after a
    ten-second simulation has already run is worse than useless.
    """

    discrete: bool = False

    # -- sampling -----------------------------------------------------------

    def sample(self, n: int, rng: Any):  # pragma: no cover - abstract
        raise NotImplementedError

    def sample_lhs(self, n: int, rng: Any):
        """Latin hypercube draw: one sample from each of ``n`` equal-probability
        strata, in random order. Same marginal, far less sampling noise on the
        mean and on the central quantiles.

        Needs :meth:`ppf`. Distributions without one fall back to plain random
        sampling, which keeps the run correct at the cost of the variance
        reduction for that one input.
        """
        np = _numpy()
        u = (rng.permutation(n) + rng.random(n)) / n
        try:
            return self.ppf(u)
        except NotImplementedError:
            return self.sample(n, rng)

    # -- analytic layer -----------------------------------------------------

    def support(self) -> Tuple[float, float]:
        return (-math.inf, math.inf)

    def pdf(self, x):
        raise NotImplementedError(f"{type(self).__name__} has no analytic density")

    def logpdf(self, x):
        np = _numpy()
        density = np.asarray(self.pdf(x), dtype=float)
        with np.errstate(divide="ignore"):
            return np.where(density > 0.0, np.log(np.where(density > 0.0, density, 1.0)), -np.inf)

    def cdf(self, x):
        raise NotImplementedError(f"{type(self).__name__} has no analytic CDF")

    def ppf(self, q):
        """Quantile function. Default: safeguarded Newton on :meth:`cdf`."""
        return _ppf_numeric(self, q)

    def moments(self) -> Moments:
        raise NotImplementedError(f"{type(self).__name__} has no closed-form moments")

    @property
    def mean_value(self) -> float:
        """The distribution's expectation (named to avoid clashing with the
        ``mean`` *parameter* of Normal and Poisson)."""
        return self.moments().mean

    @property
    def sd_value(self) -> float:
        return self.moments().sd

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(vars(self).items()) if not k.startswith("_"))
        return f"{type(self).__name__}({args})"


def _check_positive(name: str, value: float) -> float:
    if not (value > 0) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return float(value)


def _check_non_negative(name: str, value: float) -> float:
    if not (value >= 0) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return float(value)


def _check_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _as_array(x):
    return _numpy().asarray(x, dtype=float)


def _check_q(q):
    np = _numpy()
    q = np.asarray(q, dtype=float)
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("quantile levels must be in [0, 1]")
    return q


def _ppf_numeric(dist: "Distribution", q, iterations: int = 60):
    """Vectorised quantile by bracketing plus safeguarded Newton.

    Builds a bracket from the support (expanding it geometrically where the
    support is infinite), then iterates Newton steps using the density where
    one is available, bisecting whenever a step would leave the bracket.
    Converges to machine precision in well under ``iterations`` steps for
    every distribution in this module.
    """
    np = _numpy()
    q = _check_q(q)
    scalar = q.ndim == 0
    q = np.atleast_1d(q)
    lo_s, hi_s = dist.support()

    lo = np.full(q.shape, lo_s, dtype=float)
    hi = np.full(q.shape, hi_s, dtype=float)

    # Finite brackets where the support is infinite.
    try:
        m = dist.moments()
        centre = m.mean if math.isfinite(m.mean) else 0.0
        scale = m.sd if math.isfinite(m.sd) and m.sd > 0 else max(1.0, abs(centre))
    except NotImplementedError:
        centre, scale = 0.0, 1.0
    if not math.isfinite(lo_s):
        lo[:] = centre - 8.0 * scale
        for _ in range(200):
            bad = dist.cdf(lo) > q
            if not np.any(bad):
                break
            lo = np.where(bad, centre - 2.0 * (centre - lo), lo)
    if not math.isfinite(hi_s):
        hi[:] = centre + 8.0 * scale
        for _ in range(200):
            bad = dist.cdf(hi) < q
            if not np.any(bad):
                break
            hi = np.where(bad, centre + 2.0 * (hi - centre), hi)

    x = 0.5 * (lo + hi)
    has_pdf = True
    for _ in range(iterations):
        c = dist.cdf(x)
        err = c - q
        if np.all(np.abs(err) <= 1e-14) and np.all((hi - lo) <= 1e-12 * np.maximum(1.0, np.abs(x))):
            break
        lo = np.where(err < 0.0, x, lo)
        hi = np.where(err > 0.0, x, hi)
        step = None
        if has_pdf:
            try:
                density = np.asarray(dist.pdf(x), dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    step = x - err / density
            except NotImplementedError:
                has_pdf = False
        mid = 0.5 * (lo + hi)
        if step is None:
            x = mid
        else:
            inside = np.isfinite(step) & (step > lo) & (step < hi)
            x = np.where(inside, step, mid)
    # Exact hits at the boundaries of the level set.
    x = np.where(q <= 0.0, lo_s, x)
    x = np.where(q >= 1.0, hi_s, x)
    return float(x[0]) if scalar else x


class Constant(Distribution):
    """A fixed value. Useful for pinning one input while others vary."""

    def __init__(self, value: float):
        self.value = _check_finite("value", value)

    def sample(self, n, rng):
        return _numpy().full(n, self.value)

    def sample_lhs(self, n, rng):
        return self.sample(n, rng)

    def support(self):
        return (self.value, self.value)

    def cdf(self, x):
        np = _numpy()
        return np.where(_as_array(x) >= self.value, 1.0, 0.0)

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        return np.full(np.shape(q), self.value) if np.ndim(q) else self.value

    def moments(self) -> Moments:
        return Moments(self.value, 0.0)


class Uniform(Distribution):
    def __init__(self, low: float, high: float):
        if not (high > low):
            raise ValueError(f"high must exceed low, got low={low!r} high={high!r}")
        self.low, self.high = _check_finite("low", low), _check_finite("high", high)

    def sample(self, n, rng):
        return rng.uniform(self.low, self.high, n)

    def support(self):
        return (self.low, self.high)

    def pdf(self, x):
        np = _numpy()
        x = _as_array(x)
        return np.where((x >= self.low) & (x <= self.high), 1.0 / (self.high - self.low), 0.0)

    def cdf(self, x):
        np = _numpy()
        return np.clip((_as_array(x) - self.low) / (self.high - self.low), 0.0, 1.0)

    def ppf(self, q):
        return self.low + _check_q(q) * (self.high - self.low)

    def moments(self) -> Moments:
        return Moments(0.5 * (self.low + self.high), (self.high - self.low) ** 2 / 12.0)


class Normal(Distribution):
    def __init__(self, mean: float, sd: float):
        self.mean = _check_finite("mean", mean)
        self.sd = _check_non_negative("sd", sd)

    def sample(self, n, rng):
        return rng.normal(self.mean, self.sd, n)

    def pdf(self, x):
        np = _numpy()
        if self.sd == 0.0:
            return np.where(_as_array(x) == self.mean, np.inf, 0.0)
        z = (_as_array(x) - self.mean) / self.sd
        return np.exp(-0.5 * z * z) / (self.sd * math.sqrt(2.0 * math.pi))

    def cdf(self, x):
        np = _numpy()
        if self.sd == 0.0:
            return np.where(_as_array(x) >= self.mean, 1.0, 0.0)
        return _special.norm_cdf_vec((_as_array(x) - self.mean) / self.sd)

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        if self.sd == 0.0:
            return np.full(np.shape(q), self.mean) if np.ndim(q) else self.mean
        z = _special.norm_ppf_vec(q)
        out = self.mean + self.sd * z
        return float(out) if np.ndim(q) == 0 else out

    def moments(self) -> Moments:
        return Moments(self.mean, self.sd * self.sd)


class LogNormal(Distribution):
    """Log-normal in terms of the **underlying normal's** mu and sigma.

    This is the convention the C++ ``simulate_aggregate_loss`` uses, so the two
    agree. If you have a target mean and standard deviation on the natural
    scale, use :meth:`from_moments` instead — getting this backwards is the most
    common severity-modelling mistake there is.
    """

    def __init__(self, mu: float, sigma: float):
        self.mu = _check_finite("mu", mu)
        self.sigma = _check_non_negative("sigma", sigma)

    @classmethod
    def from_moments(cls, mean: float, sd: float) -> "LogNormal":
        mean = _check_positive("mean", mean)
        sd = _check_non_negative("sd", sd)
        variance = sd * sd
        sigma_sq = math.log(1.0 + variance / (mean * mean))
        return cls(mu=math.log(mean) - 0.5 * sigma_sq, sigma=math.sqrt(sigma_sq))

    @classmethod
    def from_median_cv(cls, median: float, cv: float) -> "LogNormal":
        """Median and coefficient of variation — how severity curves are often
        quoted in reinsurance submissions."""
        median = _check_positive("median", median)
        cv = _check_non_negative("cv", cv)
        return cls(mu=math.log(median), sigma=math.sqrt(math.log(1.0 + cv * cv)))

    def sample(self, n, rng):
        return rng.lognormal(self.mu, self.sigma, n)

    def support(self):
        return (0.0, math.inf)

    def pdf(self, x):
        np = _numpy()
        x = _as_array(x)
        positive = x > 0.0
        safe = np.where(positive, x, 1.0)
        if self.sigma == 0.0:
            return np.where(x == math.exp(self.mu), np.inf, 0.0)
        z = (np.log(safe) - self.mu) / self.sigma
        out = np.exp(-0.5 * z * z) / (safe * self.sigma * math.sqrt(2.0 * math.pi))
        return np.where(positive, out, 0.0)

    def cdf(self, x):
        np = _numpy()
        x = _as_array(x)
        positive = x > 0.0
        safe = np.where(positive, x, 1.0)
        if self.sigma == 0.0:
            return np.where(x >= math.exp(self.mu), 1.0, 0.0)
        out = _special.norm_cdf_vec((np.log(safe) - self.mu) / self.sigma)
        return np.where(positive, out, 0.0)

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        if self.sigma == 0.0:
            value = math.exp(self.mu)
            return np.full(np.shape(q), value) if np.ndim(q) else value
        out = np.exp(self.mu + self.sigma * _special.norm_ppf_vec(q))
        return float(out) if np.ndim(q) == 0 else out

    def moments(self) -> Moments:
        s2 = self.sigma * self.sigma
        mean = math.exp(self.mu + 0.5 * s2)
        return Moments(mean, (math.exp(s2) - 1.0) * mean * mean)


class Triangular(Distribution):
    """Low / most-likely / high. The estimator's distribution when all you have
    is a best guess and two bounds."""

    def __init__(self, low: float, mode: float, high: float):
        if not (low <= mode <= high):
            raise ValueError(f"need low <= mode <= high, got {low!r}, {mode!r}, {high!r}")
        if low == high:
            raise ValueError("low and high must differ")
        self.low, self.mode, self.high = float(low), float(mode), float(high)

    def sample(self, n, rng):
        return rng.triangular(self.low, self.mode, self.high, n)

    def support(self):
        return (self.low, self.high)

    def pdf(self, x):
        np = _numpy()
        x = _as_array(x)
        a, c, b = self.low, self.mode, self.high
        left = np.where(c > a, 2.0 * (x - a) / ((b - a) * (c - a)), 0.0)
        right = np.where(b > c, 2.0 * (b - x) / ((b - a) * (b - c)), 0.0)
        return np.where((x >= a) & (x < c), left, np.where((x >= c) & (x <= b), right, 0.0))

    def cdf(self, x):
        np = _numpy()
        x = _as_array(x)
        a, c, b = self.low, self.mode, self.high
        left = np.where(c > a, (x - a) ** 2 / ((b - a) * (c - a)), 0.0)
        right = np.where(b > c, 1.0 - (b - x) ** 2 / ((b - a) * (b - c)), 1.0)
        out = np.where(x < c, left, right)
        return np.clip(np.where(x <= a, 0.0, np.where(x >= b, 1.0, out)), 0.0, 1.0)

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        a, c, b = self.low, self.mode, self.high
        fc = (c - a) / (b - a)
        left = a + np.sqrt(q * (b - a) * (c - a))
        right = b - np.sqrt((1.0 - q) * (b - a) * (b - c))
        out = np.where(q < fc, left, right)
        return float(out) if np.ndim(q) == 0 else out

    def moments(self) -> Moments:
        a, c, b = self.low, self.mode, self.high
        return Moments((a + b + c) / 3.0, (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0)


class PERT(Distribution):
    """Beta-PERT — a smoother Triangular, standard in project and reserve risk.

    ``lam`` is the confidence placed on the mode; 4 is the classic PERT value.
    """

    def __init__(self, low: float, mode: float, high: float, lam: float = 4.0):
        if not (low <= mode <= high):
            raise ValueError(f"need low <= mode <= high, got {low!r}, {mode!r}, {high!r}")
        if low == high:
            raise ValueError("low and high must differ")
        self.low, self.mode, self.high = float(low), float(mode), float(high)
        self.lam = _check_non_negative("lam", lam)

    def _ab(self) -> Tuple[float, float]:
        span = self.high - self.low
        return (1.0 + self.lam * (self.mode - self.low) / span,
                1.0 + self.lam * (self.high - self.mode) / span)

    def sample(self, n, rng):
        a, b = self._ab()
        return self.low + rng.beta(a, b, n) * (self.high - self.low)

    def support(self):
        return (self.low, self.high)

    def pdf(self, x):
        a, b = self._ab()
        span = self.high - self.low
        return Beta(a, b).pdf((_as_array(x) - self.low) / span) / span

    def cdf(self, x):
        a, b = self._ab()
        return Beta(a, b).cdf((_as_array(x) - self.low) / (self.high - self.low))

    def ppf(self, q):
        a, b = self._ab()
        return self.low + Beta(a, b).ppf(q) * (self.high - self.low)

    def moments(self) -> Moments:
        a, b = self._ab()
        span = self.high - self.low
        m = Beta(a, b).moments()
        return Moments(self.low + m.mean * span, m.variance * span * span)


class Exponential(Distribution):
    def __init__(self, scale: float):
        self.scale = _check_positive("scale", scale)

    def sample(self, n, rng):
        return rng.exponential(self.scale, n)

    def support(self):
        return (0.0, math.inf)

    def pdf(self, x):
        np = _numpy()
        x = _as_array(x)
        return np.where(x >= 0.0, np.exp(-x / self.scale) / self.scale, 0.0)

    def cdf(self, x):
        np = _numpy()
        x = _as_array(x)
        return np.where(x >= 0.0, -np.expm1(-x / self.scale), 0.0)

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        out = -self.scale * np.log1p(-q)
        return float(out) if np.ndim(q) == 0 else out

    def moments(self) -> Moments:
        return Moments(self.scale, self.scale * self.scale)


class Gamma(Distribution):
    """Gamma with ``shape`` (k) and ``scale`` (θ): mean kθ, variance kθ²."""

    def __init__(self, shape: float, scale: float):
        self.shape = _check_positive("shape", shape)
        self.scale = _check_positive("scale", scale)

    @classmethod
    def from_moments(cls, mean: float, sd: float) -> "Gamma":
        mean = _check_positive("mean", mean)
        sd = _check_positive("sd", sd)
        return cls(shape=(mean / sd) ** 2, scale=sd * sd / mean)

    def sample(self, n, rng):
        return rng.gamma(self.shape, self.scale, n)

    def support(self):
        return (0.0, math.inf)

    def pdf(self, x):
        np = _numpy()
        x = _as_array(x)
        positive = x > 0.0
        safe = np.where(positive, x, 1.0)
        log_density = ((self.shape - 1.0) * np.log(safe) - safe / self.scale
                       - math.lgamma(self.shape) - self.shape * math.log(self.scale))
        return np.where(positive, np.exp(log_density), 0.0)

    def cdf(self, x):
        np = _numpy()
        x = _as_array(x)
        return _special.gammainc_vec(self.shape, np.maximum(x, 0.0) / self.scale)

    def sf(self, x):
        """Survival function ``P(X > x)``, computed directly in the tail."""
        np = _numpy()
        x = _as_array(x)
        return _special.gammaincc_vec(self.shape, np.maximum(x, 0.0) / self.scale)

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        scalar = np.ndim(q) == 0
        q = np.atleast_1d(q)
        k = self.shape
        safe_q = np.clip(q, 1e-300, 1.0 - 1e-16)

        # Two starting guesses, because neither is good everywhere.
        # Wilson–Hilferty (a cube-root normal approximation) is excellent in the
        # body and in the upper tail. It collapses in the lower tail, where the
        # series P(k, x) ≈ x^k / Γ(k+1) inverts directly — and that is the
        # regime where a shape below one puts the answer near 1e-30, far past
        # anything a bisection from 1 would ever reach.
        z = _special.norm_ppf_vec(safe_q)
        base = 1.0 - 1.0 / (9.0 * k) + z * math.sqrt(1.0 / (9.0 * k))
        wilson = k * np.where(base > 0.0, base, 0.0) ** 3
        with np.errstate(divide="ignore"):
            series = np.exp((np.log(safe_q) + math.lgamma(k + 1.0)) / k)
        guess = _pick_guess(self, safe_q, [wilson, series], lo=0.0, hi=math.inf)

        x = _newton_ppf(self, q, guess * self.scale, 0.0, math.inf, sf=self.sf)
        x = np.where(q <= 0.0, 0.0, np.where(q >= 1.0, np.inf, x))
        return float(x[0]) if scalar else x

    def moments(self) -> Moments:
        return Moments(self.shape * self.scale, self.shape * self.scale * self.scale)


def _pick_guess(dist, q, candidates, lo: float, hi: float):
    """Choose, elementwise, whichever starting guess already has the smallest
    CDF error. A CDF evaluation costs the same for one point as for all of
    them, so trying three approximations up front is cheaper than letting a bad
    one drag the whole batch through extra Newton passes."""
    np = _numpy()
    best = None
    best_err = None
    for candidate in candidates:
        c = np.asarray(candidate, dtype=float)
        c = np.where(np.isfinite(c) & (c > lo) & (c < hi), c, np.nan)
        with np.errstate(invalid="ignore"):
            err = np.abs(np.asarray(dist.cdf(np.nan_to_num(c, nan=0.5 * (lo + min(hi, lo + 2.0)))), dtype=float) - q)
        err = np.where(np.isnan(c), np.inf, err)
        if best is None:
            best, best_err = c, err
        else:
            take = err < best_err
            best = np.where(take, c, best)
            best_err = np.where(take, err, best_err)
    return best


def _newton_ppf(dist, q, guess, lo_s, hi_s, iterations: int = 80, tol: float = 1e-13, sf=None):
    """Safeguarded Newton inversion of a CDF, vectorised over ``q``.

    Newton doubles the correct digits each step, so a decent starting guess —
    Wilson–Hilferty for the gamma, an inflated normal quantile for the t —
    lands in four or five passes. Every step is checked against a bracket that
    tightens as it goes, and a step that would leave the bracket becomes a
    bisection instead, so it cannot diverge the way plain Newton does where the
    density collapses.

    Only the points that have not converged are carried into the next pass.
    That matters more than it sounds: a CDF evaluation costs the same whether
    one point needs it or all of them, so without the shrinking active set
    every trial would pay for the slowest trial in the batch.

    ``sf`` is the survival function, if the distribution can compute one
    directly. Above the median the residual is then ``(1 - q) - sf(x)``, in
    which every quantity is small and exactly representable, instead of
    ``cdf(x) - q`` with both terms within 1e-16 of one and the difference
    made of rounding. That is what keeps the 99.9999th percentile honest.
    """
    np = _numpy()
    q = np.asarray(q, dtype=float)
    lo = np.full(q.shape, lo_s, dtype=float)
    hi = np.full(q.shape, hi_s, dtype=float)

    x = np.array(guess, dtype=float, copy=True)
    with np.errstate(invalid="ignore"):
        seed = np.where(np.isfinite(hi), 0.5 * (np.where(np.isfinite(lo), lo, hi - 2.0) + hi),
                        np.where(np.isfinite(lo), lo + 1.0, 0.0))
    x = np.where(np.isfinite(x) & (x > lo) & (x < hi), x, seed)

    active = np.ones(q.shape, dtype=bool)
    for _ in range(iterations):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        xa, qa = x[idx], q[idx]
        if sf is None:
            err = np.asarray(dist.cdf(xa), dtype=float) - qa
        else:
            upper = qa > 0.5
            err = np.empty(xa.shape, dtype=float)
            if np.any(~upper):
                err[~upper] = np.asarray(dist.cdf(xa[~upper]), dtype=float) - qa[~upper]
            if np.any(upper):
                err[upper] = (1.0 - qa[upper]) - np.asarray(sf(xa[upper]), dtype=float)
        la = np.where(err < 0.0, np.maximum(lo[idx], xa), lo[idx])
        ha = np.where(err > 0.0, np.minimum(hi[idx], xa), hi[idx])
        lo[idx], hi[idx] = la, ha

        density = np.asarray(dist.pdf(xa), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            step = xa - err / density
        good = np.isfinite(step) & (step > la) & (step < ha)

        # Where Newton escaped: bisect a finite bracket, or march outward to
        # find one. Marching costs a CDF call, which is why it is the fallback.
        both = np.isfinite(la) & np.isfinite(ha)
        with np.errstate(invalid="ignore"):
            mid = np.where(both, 0.5 * (la + ha), xa)
        mid = np.where(both | good, mid,
                       np.where(np.isfinite(ha), np.minimum(2.0 * xa - ha, ha) - 1.0,
                                np.where(xa > 0.0, 2.0 * xa, xa + 1.0) + 1.0))
        nxt = np.where(good, step, mid)

        # A point whose CDF error is already at the noise floor keeps the x it
        # has. Overwriting it with the fallback bisection — which is what the
        # step becomes when the density underflows — and then calling it
        # converged is how this loop silently returns the wrong quantile.
        # The CDF error worth stopping at is relative to the tail mass being
        # resolved — a ten-trillionth of q = 1e-7 — never to the larger side,
        # which near q = 0 would let the solver quit a million times too
        # early. Where that is below what the CDF can resolve, the step-size
        # test below ends the iteration instead; no absolute floor is needed.
        tiny = np.abs(err) <= tol * np.minimum(qa, 1.0 - qa)
        # Relative to the value itself, never to an absolute floor: the lower
        # quantiles of a shape < 1 gamma live around 1e-15, where "the step was
        # smaller than 1e-14" means "we stopped ten times away from the answer".
        moved = np.abs(nxt - xa) > 1e-14 * np.maximum(np.abs(xa), np.abs(nxt))
        x[idx] = np.where(tiny, xa, nxt)
        active[idx] = ~tiny & moved
    return x


class Beta(Distribution):
    def __init__(self, a: float, b: float):
        self.a = _check_positive("a", a)
        self.b = _check_positive("b", b)

    def sample(self, n, rng):
        return rng.beta(self.a, self.b, n)

    def support(self):
        return (0.0, 1.0)

    def pdf(self, x):
        np = _numpy()
        x = _as_array(x)
        inside = (x > 0.0) & (x < 1.0)
        safe = np.where(inside, x, 0.5)
        log_density = ((self.a - 1.0) * np.log(safe) + (self.b - 1.0) * np.log1p(-safe)
                       - (math.lgamma(self.a) + math.lgamma(self.b) - math.lgamma(self.a + self.b)))
        out = np.where(inside, np.exp(log_density), 0.0)
        # Endpoints: density can be finite and non-zero when a or b == 1.
        out = np.where((x == 0.0) & (self.a == 1.0), self.b, out)
        out = np.where((x == 1.0) & (self.b == 1.0), self.a, out)
        return out

    def cdf(self, x):
        np = _numpy()
        return _special.betainc_vec(self.a, self.b, np.clip(_as_array(x), 0.0, 1.0))

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        scalar = np.ndim(q) == 0
        q = np.atleast_1d(q)

        # Above the median, solve the mirror problem: X ~ Beta(a, b) means
        # 1 - X ~ Beta(b, a), and the upper quantile of one is one minus the
        # lower quantile of the other. The lower tail is where the series
        # guess and the CDF keep their digits; the upper tail, solved
        # directly, is where they do not.
        upper = q > 0.5
        out = np.empty(q.shape, dtype=float)
        if np.any(~upper):
            out[~upper] = self._ppf_lower(q[~upper])
        if np.any(upper):
            out[upper] = 1.0 - Beta(self.b, self.a)._ppf_lower(1.0 - q[upper])
        return float(out[0]) if scalar else out

    def _ppf_lower(self, q):
        """Quantiles for ``q <= 0.5`` — see :meth:`ppf` for why only those."""
        np = _numpy()
        safe_q = np.clip(q, 1e-300, 1.0 - 1e-16)
        m = self.moments()
        log_beta = math.lgamma(self.a) + math.lgamma(self.b) - math.lgamma(self.a + self.b)

        # Body, then each tail from its own series. With a shape below one the
        # true quantile can sit at 1e-37, which no bisection of [0, 1] reaches
        # in any number of steps a person would wait for — but
        # I_x(a, b) ≈ x^a / (a·B(a, b)) inverts to it in closed form.
        body = np.clip(m.mean + m.sd * _special.norm_ppf_vec(safe_q), 1e-12, 1.0 - 1e-12)
        with np.errstate(divide="ignore"):
            lower = np.exp((np.log(safe_q) + math.log(self.a) + log_beta) / self.a)
            upper = -np.expm1((np.log1p(-safe_q) + math.log(self.b) + log_beta) / self.b)
        guess = _pick_guess(self, safe_q, [body, lower, upper], lo=0.0, hi=1.0)

        x = _newton_ppf(self, q, guess, 0.0, 1.0)
        return np.where(q <= 0.0, 0.0, np.where(q >= 1.0, 1.0, x))

    def moments(self) -> Moments:
        a, b = self.a, self.b
        return Moments(a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1.0)))


class Pareto(Distribution):
    """Pareto Type I with scale ``xm`` and tail index ``alpha``.

    Matches the C++ catastrophe simulator. Note the mean is infinite for
    ``alpha <= 1`` and the variance for ``alpha <= 2`` — a sample mean will
    still print a number, and it will be meaningless. :meth:`moments` reports
    ``inf`` in those cases rather than a number.
    """

    def __init__(self, xm: float, alpha: float):
        self.xm = _check_positive("xm", xm)
        self.alpha = _check_positive("alpha", alpha)

    def sample(self, n, rng):
        return self.xm * (1.0 + rng.pareto(self.alpha, n))

    def support(self):
        return (self.xm, math.inf)

    def pdf(self, x):
        np = _numpy()
        x = _as_array(x)
        inside = x >= self.xm
        safe = np.where(inside, x, self.xm)
        return np.where(inside, self.alpha * self.xm ** self.alpha / safe ** (self.alpha + 1.0), 0.0)

    def cdf(self, x):
        np = _numpy()
        x = _as_array(x)
        inside = x >= self.xm
        safe = np.where(inside, x, self.xm)
        return np.where(inside, 1.0 - (self.xm / safe) ** self.alpha, 0.0)

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        with np.errstate(divide="ignore"):
            out = self.xm * (1.0 - q) ** (-1.0 / self.alpha)
        return float(out) if np.ndim(q) == 0 else out

    def moments(self) -> Moments:
        a, xm = self.alpha, self.xm
        mean = a * xm / (a - 1.0) if a > 1.0 else math.inf
        var = xm * xm * a / ((a - 1.0) ** 2 * (a - 2.0)) if a > 2.0 else math.inf
        return Moments(mean, var)


class Weibull(Distribution):
    """Weibull with ``shape`` (k) and ``scale`` (λ). ``k < 1`` gives a heavier
    tail than exponential, ``k > 1`` a lighter one — useful as the middle
    ground between Gamma and Pareto for severity curves."""

    def __init__(self, shape: float, scale: float):
        self.shape = _check_positive("shape", shape)
        self.scale = _check_positive("scale", scale)

    def sample(self, n, rng):
        return self.scale * rng.weibull(self.shape, n)

    def support(self):
        return (0.0, math.inf)

    def pdf(self, x):
        np = _numpy()
        x = _as_array(x)
        positive = x > 0.0
        z = np.where(positive, x, 1.0) / self.scale
        out = (self.shape / self.scale) * z ** (self.shape - 1.0) * np.exp(-(z ** self.shape))
        return np.where(positive, out, 0.0)

    def cdf(self, x):
        np = _numpy()
        x = _as_array(x)
        return np.where(x > 0.0, -np.expm1(-(np.maximum(x, 0.0) / self.scale) ** self.shape), 0.0)

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        out = self.scale * (-np.log1p(-q)) ** (1.0 / self.shape)
        return float(out) if np.ndim(q) == 0 else out

    def moments(self) -> Moments:
        g1 = math.gamma(1.0 + 1.0 / self.shape)
        g2 = math.gamma(1.0 + 2.0 / self.shape)
        return Moments(self.scale * g1, self.scale * self.scale * (g2 - g1 * g1))


class StudentT(Distribution):
    """Student-t with location and scale: the standard fat-tailed model for
    returns. ``df`` need not be an integer; below 2 the variance is infinite,
    below 1 so is the mean, and :meth:`moments` says so."""

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0):
        self.df = _check_positive("df", df)
        self.loc = _check_finite("loc", loc)
        self.scale = _check_positive("scale", scale)

    def sample(self, n, rng):
        return self.loc + self.scale * rng.standard_t(self.df, n)

    def pdf(self, x):
        np = _numpy()
        t = (_as_array(x) - self.loc) / self.scale
        nu = self.df
        log_norm = math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) - 0.5 * math.log(nu * math.pi)
        return np.exp(log_norm - 0.5 * (nu + 1.0) * np.log1p(t * t / nu)) / self.scale

    def cdf(self, x):
        np = _numpy()
        t = (_as_array(x) - self.loc) / self.scale
        nu = self.df
        ib = _special.betainc_vec(0.5 * nu, 0.5, nu / (nu + t * t))
        return np.where(t > 0.0, 1.0 - 0.5 * ib, 0.5 * ib)

    def ppf(self, q):
        """Quantile by inverting the beta, not by Newton on the t itself.

        ``F(t) = ½·I_{ν/(ν+t²)}(ν/2, ½)`` for ``t < 0``, so the t quantile is an
        exact algebraic function of a beta quantile. That inherits the beta's
        tail handling, which matters here more than anywhere else: with two or
        three degrees of freedom the tail is polynomial, and a solver working
        directly in ``t`` has to travel a very long way to find it.
        """
        np = _numpy()
        q = _check_q(q)
        scalar = np.ndim(q) == 0
        q = np.atleast_1d(q)
        nu = self.df

        tail = np.minimum(q, 1.0 - q)
        # Two routes, each exact where the other is not. Far out, invert
        # I_z(ν/2, ½) = 2·tail directly. Near the median that puts z at
        # 1 − 1e-14 and the needed quantity 1 − z has two digits left; so
        # there invert the complementary beta for w = 1 − z instead, where
        # the small number is the *input* (1 − 2·tail) and is exact.
        far = tail < 0.25
        z = Beta(0.5 * nu, 0.5).ppf(np.clip(2.0 * np.where(far, tail, 0.1), 1e-300, 1.0))
        w = Beta(0.5, 0.5 * nu).ppf(np.clip(1.0 - 2.0 * np.where(far, 0.1, tail), 0.0, 1.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            from_z = np.sqrt(nu * (1.0 - z) / z)
            from_w = np.sqrt(nu * w / (1.0 - w))
        magnitude = np.where(far, np.where(z > 0.0, from_z, np.inf), from_w)
        t = np.where(q < 0.5, -magnitude, magnitude)
        t = np.where(q == 0.5, 0.0, t)

        x = self.loc + self.scale * t
        x = np.where(q <= 0.0, -np.inf, np.where(q >= 1.0, np.inf, x))
        return float(x[0]) if scalar else x

    def moments(self) -> Moments:
        mean = self.loc if self.df > 1.0 else math.inf
        var = self.scale ** 2 * self.df / (self.df - 2.0) if self.df > 2.0 else math.inf
        return Moments(mean, var)


# -- discrete ---------------------------------------------------------------


class _Discrete(Distribution):
    """Counting distributions on {0, 1, 2, …}. CDF and quantiles come from the
    cumulative sum of the pmf, which is exact and vectorises."""

    discrete = True

    def pmf(self, k):  # pragma: no cover - abstract
        raise NotImplementedError

    def pdf(self, x):
        """The pmf, evaluated at integer ``x`` (zero elsewhere)."""
        np = _numpy()
        x = _as_array(x)
        integer = np.isclose(x, np.round(x))
        return np.where(integer, self.pmf(np.round(x)), 0.0)

    def support(self):
        return (0.0, math.inf)

    def _upper(self) -> int:
        m = self.moments()
        if not math.isfinite(m.variance):
            return 10_000_000
        return int(m.mean + 40.0 * m.sd + 50.0)

    def _table(self, upto: int):
        np = _numpy()
        k = np.arange(0, int(upto) + 1, dtype=float)
        p = self.pmf(k)
        return k, p, np.cumsum(p)

    def cdf(self, x):
        np = _numpy()
        x = _as_array(x)
        floor = np.floor(x)
        top = int(min(max(float(np.max(floor)) if floor.size else 0.0, 0.0), self._upper()))
        _, _, cumulative = self._table(top)
        idx = np.clip(floor, -1, top).astype(int)
        out = np.where(idx >= 0, cumulative[np.maximum(idx, 0)], 0.0)
        return np.clip(np.where(floor > top, 1.0, out), 0.0, 1.0)

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        scalar = np.ndim(q) == 0
        q = np.atleast_1d(q)
        k, _, cumulative = self._table(self._upper())
        # Smallest k with F(k) >= q. Round-off at the top is absorbed by clip.
        idx = np.searchsorted(cumulative, np.minimum(q, cumulative[-1]), side="left")
        out = k[np.clip(idx, 0, k.size - 1)]
        return float(out[0]) if scalar else out


class Poisson(_Discrete):
    def __init__(self, mean: float):
        self.mean = _check_non_negative("mean", mean)

    def sample(self, n, rng):
        return rng.poisson(self.mean, n).astype(float)

    def pmf(self, k):
        np = _numpy()
        k = _as_array(k)
        if self.mean == 0.0:
            return np.where(k == 0.0, 1.0, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = k * math.log(self.mean) - self.mean - np.vectorize(math.lgamma)(k + 1.0)
        return np.where(k >= 0.0, np.exp(log_p), 0.0)

    def moments(self) -> Moments:
        return Moments(self.mean, self.mean)


class NegativeBinomial(_Discrete):
    """Over-dispersed count model — claim frequency when Poisson is too tidy.

    Parameterised by ``n_success`` (r, the shape, need not be an integer) and
    ``p``: mean r(1-p)/p and variance r(1-p)/p². Use
    :meth:`from_mean_dispersion` when you know the mean and the
    variance-to-mean ratio, which is how frequency is usually quoted.
    """

    def __init__(self, n_success: float, p: float):
        self.n_success = _check_positive("n_success", n_success)
        if not (0.0 < p <= 1.0):
            raise ValueError(f"p must be in (0, 1], got {p!r}")
        self.p = float(p)

    @classmethod
    def from_mean_dispersion(cls, mean: float, dispersion: float) -> "NegativeBinomial":
        """Parameterise by mean and variance-to-mean ratio (must exceed 1)."""
        mean = _check_positive("mean", mean)
        if not (dispersion > 1.0):
            raise ValueError(f"dispersion must be > 1 (else use Poisson), got {dispersion!r}")
        p = 1.0 / dispersion
        return cls(n_success=mean * p / (1.0 - p), p=p)

    def sample(self, n, rng):
        return rng.negative_binomial(self.n_success, self.p, n).astype(float)

    def pmf(self, k):
        np = _numpy()
        k = _as_array(k)
        r, p = self.n_success, self.p
        lg = np.vectorize(math.lgamma)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = (lg(k + r) - lg(r) - lg(k + 1.0) + r * math.log(p)
                     + k * (math.log1p(-p) if p < 1.0 else -math.inf))
            log_p = np.where((k == 0.0) & (p >= 1.0), 0.0, log_p)
        return np.where(k >= 0.0, np.exp(log_p), 0.0)

    def moments(self) -> Moments:
        r, p = self.n_success, self.p
        return Moments(r * (1.0 - p) / p, r * (1.0 - p) / (p * p))


class Bernoulli(_Discrete):
    def __init__(self, p: float):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p!r}")
        self.p = float(p)

    def sample(self, n, rng):
        return (rng.random(n) < self.p).astype(float)

    def pmf(self, k):
        np = _numpy()
        k = _as_array(k)
        return np.where(k == 1.0, self.p, np.where(k == 0.0, 1.0 - self.p, 0.0))

    def support(self):
        return (0.0, 1.0)

    def _upper(self) -> int:
        return 1

    def moments(self) -> Moments:
        return Moments(self.p, self.p * (1.0 - self.p))


class Binomial(_Discrete):
    def __init__(self, trials: int, p: float):
        if trials < 0:
            raise ValueError(f"trials must be >= 0, got {trials!r}")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p!r}")
        self.trials, self.p = int(trials), float(p)

    def sample(self, n, rng):
        return rng.binomial(self.trials, self.p, n).astype(float)

    def pmf(self, k):
        np = _numpy()
        k = _as_array(k)
        m, p = self.trials, self.p
        valid = (k >= 0.0) & (k <= m)
        safe = np.where(valid, k, 0.0)
        lg = np.vectorize(math.lgamma)
        if p == 0.0:
            return np.where(k == 0.0, 1.0, 0.0)
        if p == 1.0:
            return np.where(k == float(m), 1.0, 0.0)
        log_p = (lg(m + 1.0) - lg(safe + 1.0) - lg(m - safe + 1.0)
                 + safe * math.log(p) + (m - safe) * math.log1p(-p))
        return np.where(valid, np.exp(log_p), 0.0)

    def support(self):
        return (0.0, float(self.trials))

    def _upper(self) -> int:
        return self.trials

    def moments(self) -> Moments:
        return Moments(self.trials * self.p, self.trials * self.p * (1.0 - self.p))


# -- data-driven and composite ------------------------------------------------


class Empirical(Distribution):
    """Resample your own history with replacement — the bootstrap.

    Often the honest choice: it makes no distributional assumption at all, at
    the cost of never producing a loss larger than the worst one you have seen.
    """

    def __init__(self, data: Sequence[float]):
        np = _numpy()
        values = np.asarray(list(data), dtype=float)
        if values.size == 0:
            raise ValueError("Empirical needs at least one observation")
        if not np.all(np.isfinite(values)):
            raise ValueError("Empirical data must be finite")
        self.data = values
        self._sorted = np.sort(values)

    def sample(self, n, rng):
        return rng.choice(self.data, size=n, replace=True)

    def support(self):
        return (float(self._sorted[0]), float(self._sorted[-1]))

    def cdf(self, x):
        np = _numpy()
        return np.searchsorted(self._sorted, _as_array(x), side="right") / self._sorted.size

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        idx = np.clip(np.ceil(q * self._sorted.size).astype(int) - 1, 0, self._sorted.size - 1)
        out = self._sorted[idx]
        return float(out) if np.ndim(q) == 0 else out

    def moments(self) -> Moments:
        return Moments(float(self.data.mean()), float(self.data.var()))

    def __repr__(self) -> str:
        return f"Empirical(n={self.data.size})"


class Categorical(Distribution):
    """A finite set of outcomes with probabilities — scenario weights, rating
    classes, discrete severity tables."""

    discrete = True

    def __init__(self, values: Sequence[float], probs: Optional[Sequence[float]] = None):
        np = _numpy()
        v = np.asarray(list(values), dtype=float)
        if v.size == 0:
            raise ValueError("Categorical needs at least one value")
        if probs is None:
            p = np.full(v.size, 1.0 / v.size)
        else:
            p = np.asarray(list(probs), dtype=float)
            if p.shape != v.shape:
                raise ValueError("probs must have one entry per value")
            if np.any(p < 0.0) or not np.all(np.isfinite(p)):
                raise ValueError("probs must be finite and >= 0")
            total = p.sum()
            if total <= 0.0:
                raise ValueError("probs must have positive total mass")
            p = p / total
        order = np.argsort(v, kind="mergesort")
        self.values = v[order]
        self.probs = p[order]
        self._cumulative = np.cumsum(self.probs)

    def sample(self, n, rng):
        return rng.choice(self.values, size=n, p=self.probs)

    def support(self):
        return (float(self.values[0]), float(self.values[-1]))

    def pmf(self, k):
        np = _numpy()
        k = _as_array(k)
        idx = np.searchsorted(self.values, k)
        idx = np.clip(idx, 0, self.values.size - 1)
        return np.where(self.values[idx] == k, self.probs[idx], 0.0)

    def pdf(self, x):
        return self.pmf(x)

    def cdf(self, x):
        np = _numpy()
        idx = np.searchsorted(self.values, _as_array(x), side="right")
        return np.where(idx > 0, self._cumulative[np.maximum(idx - 1, 0)], 0.0)

    def ppf(self, q):
        np = _numpy()
        q = _check_q(q)
        idx = np.searchsorted(self._cumulative, np.minimum(q, self._cumulative[-1]), side="left")
        out = self.values[np.clip(idx, 0, self.values.size - 1)]
        return float(out) if np.ndim(q) == 0 else out

    def moments(self) -> Moments:
        mean = float((self.values * self.probs).sum())
        return Moments(mean, float(((self.values - mean) ** 2 * self.probs).sum()))

    def __repr__(self) -> str:
        return f"Categorical(n={self.values.size})"


class Mixture(Distribution):
    """A weighted mixture of distributions: attritional plus large losses,
    a two-regime economy, a spliced severity curve.

    Weights are normalised. Sampling draws the component count from a
    multinomial and samples each component once, so it stays vectorised.
    """

    def __init__(self, components: Sequence[Distribution], weights: Optional[Sequence[float]] = None):
        np = _numpy()
        comps = list(components)
        if not comps:
            raise ValueError("Mixture needs at least one component")
        for c in comps:
            if not isinstance(c, Distribution):
                raise TypeError(f"components must be Distributions, got {type(c).__name__}")
        if weights is None:
            w = np.full(len(comps), 1.0 / len(comps))
        else:
            w = np.asarray(list(weights), dtype=float)
            if w.size != len(comps):
                raise ValueError("one weight per component")
            if np.any(w < 0.0) or not np.all(np.isfinite(w)) or w.sum() <= 0.0:
                raise ValueError("weights must be finite, >= 0 and not all zero")
            w = w / w.sum()
        self.components = comps
        self.weights = w
        self.discrete = all(c.discrete for c in comps)

    def sample(self, n, rng):
        np = _numpy()
        counts = rng.multinomial(n, self.weights)
        out = np.empty(n, dtype=float)
        start = 0
        for comp, count in zip(self.components, counts):
            if count:
                out[start:start + count] = comp.sample(int(count), rng)
                start += count
        rng.shuffle(out)
        return out

    def support(self):
        lo = min(c.support()[0] for c in self.components)
        hi = max(c.support()[1] for c in self.components)
        return (lo, hi)

    def pdf(self, x):
        return sum(w * c.pdf(x) for w, c in zip(self.weights, self.components))

    def cdf(self, x):
        return sum(w * c.cdf(x) for w, c in zip(self.weights, self.components))

    def moments(self) -> Moments:
        ms = [c.moments() for c in self.components]
        mean = sum(w * m.mean for w, m in zip(self.weights, ms))
        second = sum(w * (m.variance + m.mean * m.mean) for w, m in zip(self.weights, ms))
        return Moments(mean, second - mean * mean)

    def __repr__(self) -> str:
        parts = ", ".join(f"{w:.3g}×{c!r}" for w, c in zip(self.weights, self.components))
        return f"Mixture({parts})"


class Truncated(Distribution):
    """A distribution conditioned on ``low <= X <= high`` — a severity curve
    that cannot exceed the policy limit, a rate that cannot go negative.

    Sampling is by inversion on the truncated probability range, so it never
    rejects and never loops. Moments are integrated numerically (Simpson on a
    fine grid, accurate to about 1e-8 relative); the docstring says so because
    a user checking against a closed form will otherwise wonder.
    """

    def __init__(self, dist: Distribution, low: Optional[float] = None, high: Optional[float] = None):
        if not isinstance(dist, Distribution):
            raise TypeError("dist must be a Distribution")
        if low is None and high is None:
            raise ValueError("give at least one of low or high")
        s_lo, s_hi = dist.support()
        self.dist = dist
        self.low = s_lo if low is None else float(low)
        self.high = s_hi if high is None else float(high)
        if not (self.high > self.low):
            raise ValueError("high must exceed low")
        self._f_low = float(dist.cdf(self.low)) if math.isfinite(self.low) else 0.0
        self._f_high = float(dist.cdf(self.high)) if math.isfinite(self.high) else 1.0
        # Left-closed: include the mass at `low` itself for discrete parents.
        if dist.discrete and math.isfinite(self.low):
            self._f_low = float(dist.cdf(self.low - 1.0))
        self._mass = self._f_high - self._f_low
        if not (self._mass > 0.0):
            raise ValueError("the truncation interval carries no probability mass")
        self.discrete = dist.discrete

    def sample(self, n, rng):
        u = self._f_low + self._mass * rng.random(n)
        return self.dist.ppf(u)

    def support(self):
        return (self.low, self.high)

    def pdf(self, x):
        np = _numpy()
        x = _as_array(x)
        inside = (x >= self.low) & (x <= self.high)
        return np.where(inside, self.dist.pdf(x) / self._mass, 0.0)

    def cdf(self, x):
        np = _numpy()
        x = _as_array(x)
        out = (self.dist.cdf(np.clip(x, self.low, self.high)) - self._f_low) / self._mass
        return np.clip(np.where(x < self.low, 0.0, np.where(x >= self.high, 1.0, out)), 0.0, 1.0)

    def ppf(self, q):
        q = _check_q(q)
        return self.dist.ppf(self._f_low + self._mass * q)

    def moments(self) -> Moments:
        np = _numpy()
        if self.discrete:
            top = self.high if math.isfinite(self.high) else float(self.dist.ppf(1.0 - 1e-12))
            k = np.arange(math.floor(self.low), math.ceil(top) + 1, dtype=float)
            p = self.pdf(k)
            mean = float((k * p).sum())
            return Moments(mean, float(((k - mean) ** 2 * p).sum()))
        lo = self.low if math.isfinite(self.low) else float(self.dist.ppf(1e-10))
        hi = self.high if math.isfinite(self.high) else float(self.dist.ppf(1.0 - 1e-10))
        grid = np.linspace(lo, hi, 20_001)
        density = self.pdf(grid)
        weights = np.ones_like(grid)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        h = (hi - lo) / (grid.size - 1)
        mass = float((weights * density).sum() * h / 3.0)
        mean = float((weights * density * grid).sum() * h / 3.0) / mass
        var = float((weights * density * (grid - mean) ** 2).sum() * h / 3.0) / mass
        return Moments(mean, var)

    def __repr__(self) -> str:
        return f"Truncated({self.dist!r}, low={self.low!r}, high={self.high!r})"


# ---------------------------------------------------------------------------
# Dependence
# ---------------------------------------------------------------------------


def correlation_matrix(names: Sequence[str], pairs: Mapping[Tuple[str, str], float]):
    """Assemble a full correlation matrix from ``{(a, b): rho}`` pairs.

    Unspecified pairs are independent (zero). The matrix is checked for
    symmetry-by-construction and positive definiteness; if it is not positive
    definite the error names the offending combination rather than letting a
    Cholesky failure surface later.
    """
    np = _numpy()
    index = {name: i for i, name in enumerate(names)}
    matrix = np.eye(len(names))
    for (a, b), rho in pairs.items():
        if a not in index or b not in index:
            unknown = a if a not in index else b
            raise ValueError(f"correlation refers to unknown variable {unknown!r}; known: {list(names)}")
        if a == b:
            raise ValueError(f"cannot set a variable's correlation with itself ({a!r})")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"correlation must be in [-1, 1], got {rho!r} for {(a, b)!r}")
        matrix[index[a], index[b]] = matrix[index[b], index[a]] = float(rho)
    _check_correlation(matrix)
    return matrix


def _check_correlation(matrix) -> None:
    np = _numpy()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"correlation must be a square matrix, got shape {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-12):
        raise ValueError("correlation matrix must have ones on the diagonal")
    eigen = np.linalg.eigvalsh(matrix)
    if eigen.min() <= -1e-10:
        raise ValueError(
            f"correlation matrix is not positive semi-definite (smallest eigenvalue "
            f"{eigen.min():.3g}). The pairwise correlations you asked for cannot all "
            f"hold at once — reduce the strongest ones, or pass the matrix through "
            f"riskpy.capital.nearest_correlation()."
        )


def iman_conover(samples: Dict[str, Any], target, rng, spearman: bool = True) -> Dict[str, Any]:
    """Impose a rank correlation structure on independently drawn samples.

    Iman & Conover (1982): reorder each column so its ranks follow those of a
    correlated multivariate normal score matrix. The marginals are untouched —
    every sampled value is kept, only the pairing changes — which is why this
    works with any distribution and with Latin hypercube draws.

    ``target`` is the Spearman rank correlation by default. Since the scores are
    normal, the Pearson correlation that produces a given Spearman value is
    ``2·sin(π·ρ_s/6)``, and that conversion is applied unless
    ``spearman=False``.
    """
    np = _numpy()
    names = list(samples)
    k = len(names)
    target = np.asarray(target, dtype=float)
    if target.shape != (k, k):
        raise ValueError(f"target must be {k}x{k} for {k} variables, got {target.shape}")
    _check_correlation(target)
    n = len(samples[names[0]])
    if n < 2:
        return dict(samples)

    pearson = 2.0 * np.sin(math.pi * target / 6.0) if spearman else target
    pearson = np.clip(pearson, -1.0, 1.0)
    np.fill_diagonal(pearson, 1.0)

    scores = _special.norm_ppf_vec(np.arange(1, n + 1) / (n + 1.0))
    R = np.column_stack([scores[rng.permutation(n)] for _ in range(k)])
    E = np.corrcoef(R, rowvar=False)
    F = np.linalg.cholesky(E)
    try:
        P = np.linalg.cholesky(pearson)
    except np.linalg.LinAlgError:
        # Semi-definite target: nudge the diagonal.
        P = np.linalg.cholesky(pearson + 1e-10 * np.eye(k))
    T = R @ np.linalg.inv(F).T @ P.T

    out = {}
    for j, name in enumerate(names):
        column = np.asarray(samples[name], dtype=float)
        order = np.argsort(T[:, j], kind="mergesort")
        ranked = np.empty(n, dtype=float)
        ranked[order] = np.sort(column)
        out[name] = ranked
    return out


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """The output of a run, plus the questions you actually ask of it.

    ``values`` is the simulated output, one entry per trial. ``inputs`` holds
    the sampled inputs (unless the run was told not to keep them), which is what
    makes sensitivity analysis possible after the fact.

    Quantile queries sort the values once and cache the order, so calling
    ``var`` at a dozen levels costs one sort, not twelve partial ones.
    """

    values: Any
    inputs: Dict[str, Any]
    trials: int
    seed: Optional[int]
    label: str = "output"
    _sorted: Any = field(default=None, repr=False, compare=False)

    # -- internals ----------------------------------------------------------

    def _ordered(self):
        if self._sorted is None:
            self._sorted = _numpy().sort(self.values)
        return self._sorted

    # -- summary statistics -------------------------------------------------

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def std(self) -> float:
        # Sample standard deviation (ddof=1): these are draws, not a population.
        return float(self.values.std(ddof=1)) if self.trials > 1 else 0.0

    @property
    def standard_error(self) -> float:
        """How much the *mean* would wobble if you re-ran with a new seed.

        The number that tells you whether you ran enough trials. Shrinks with
        the square root of ``trials``, so ten times the precision costs a
        hundred times the work.
        """
        return self.std / math.sqrt(self.trials)

    @property
    def median(self) -> float:
        return self.percentile(50.0)

    @property
    def skewness(self) -> float:
        """Sample skewness. Positive means a long right tail — the usual shape
        of a loss distribution — and its size is a warning about how badly a
        normal approximation will do."""
        np = _numpy()
        if self.std == 0.0:
            return 0.0
        z = (self.values - self.mean) / self.values.std()
        return float(np.mean(z ** 3))

    @property
    def kurtosis(self) -> float:
        """Excess kurtosis (normal = 0)."""
        np = _numpy()
        if self.std == 0.0:
            return 0.0
        z = (self.values - self.mean) / self.values.std()
        return float(np.mean(z ** 4) - 3.0)

    def percentile(self, q: float) -> float:
        """``q`` in percent, e.g. ``percentile(99.5)``. Linear interpolation
        between order statistics, identical to NumPy's default."""
        np = _numpy()
        if not (0.0 <= q <= 100.0):
            raise ValueError(f"q is a percentage in [0, 100], got {q!r}")
        ordered = self._ordered()
        n = ordered.size
        pos = (q / 100.0) * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return float(ordered[lo] + frac * (ordered[hi] - ordered[lo]))

    def percentiles(self, qs: Iterable[float]) -> Dict[float, float]:
        return {float(q): self.percentile(q) for q in qs}

    def var(self, level: float = 0.995) -> float:
        """Value at Risk — the loss exceeded with probability ``1 - level``."""
        _check_level(level)
        return self.percentile(level * 100.0)

    def tvar(self, level: float = 0.995) -> float:
        """Tail VaR / Expected Shortfall — the mean of the losses beyond VaR.

        Preferred to VaR for the same reason regulators prefer it: VaR tells you
        the threshold and nothing about how bad it gets past it.
        """
        _check_level(level)
        np = _numpy()
        ordered = self._ordered()
        threshold = self.var(level)
        start = int(np.searchsorted(ordered, threshold, side="left"))
        tail = ordered[start:]
        if tail.size == 0:  # pragma: no cover - only if every value is identical
            return threshold
        return float(tail.mean())

    def cdf(self, x: float) -> float:
        """Empirical P(output <= x)."""
        np = _numpy()
        return float(np.searchsorted(self._ordered(), x, side="right") / self.trials)

    def prob_above(self, threshold: float) -> float:
        return float((self.values > threshold).mean())

    def prob_below(self, threshold: float) -> float:
        return float((self.values < threshold).mean())

    def histogram(self, bins: int = 60):
        """``(edges, counts)`` — the data behind :func:`riskpy.viz.distribution`."""
        np = _numpy()
        counts, edges = np.histogram(self.values, bins=bins)
        return edges, counts

    def exceedance_curve(self):
        """``(x, P(output > x))`` at every observed value, for a survival plot."""
        np = _numpy()
        ordered = self._ordered()
        survival = 1.0 - np.arange(1, ordered.size + 1) / (ordered.size + 1.0)
        return ordered, survival

    # -- diagnostics --------------------------------------------------------

    def convergence(self, points: int = 200):
        """Running mean and a ±1.96 standard-error band, for a convergence plot.

        Returns ``(n, running_mean, half_width)``. If the band is still visibly
        narrowing at the right-hand edge, you have not run enough trials.
        """
        np = _numpy()
        points = max(2, min(int(points), self.trials))
        first = max(2, self.trials // points)
        idx = np.unique(np.linspace(first, self.trials, points).astype(int))
        cumulative = np.cumsum(self.values)
        running = cumulative[idx - 1] / idx
        # Running (unbiased) variance via the sum of squares — one pass, no loop.
        cumulative_sq = np.cumsum(self.values ** 2)
        variance = np.maximum((cumulative_sq[idx - 1] - idx * running ** 2) / np.maximum(idx - 1, 1), 0.0)
        half_width = 1.96 * np.sqrt(variance / idx)
        return idx, running, half_width

    def sensitivity(self, method: str = "spearman") -> List[tuple]:
        """Rank inputs by how much they move the output — the tornado data.

        Spearman rank correlation by default, not Pearson: the relationship
        between an input and a total is usually monotonic but rarely linear,
        and rank correlation does not care about the shape. ``method="pearson"``
        is there for the linear case, and ``"contribution"`` gives each input's
        share of explained variance (squared Pearson, normalised to sum to 1).

        Returns ``[(name, score), …]``, strongest first.
        """
        if not self.inputs:
            raise ValueError(
                "This run kept no inputs, so sensitivity cannot be computed. "
                "Re-run with keep_inputs=True (the default)."
            )
        if method not in ("spearman", "pearson", "contribution"):
            raise ValueError("method must be 'spearman', 'pearson' or 'contribution'")
        np = _numpy()
        if method == "spearman":
            target = _rankdata(self.values, np)
        else:
            target = self.values
        scores = []
        for name, column in self.inputs.items():
            if float(np.std(column)) == 0.0 or float(np.std(target)) == 0.0:
                scores.append((name, 0.0))  # a Constant moves nothing
                continue
            probe = _rankdata(column, np) if method == "spearman" else column
            corr = float(np.corrcoef(probe, target)[0, 1])
            scores.append((name, 0.0 if math.isnan(corr) else corr))
        if method == "contribution":
            total = sum(c * c for _, c in scores)
            scores = [(name, (c * c / total if total > 0 else 0.0)) for name, c in scores]
        scores.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return scores

    # -- presentation -------------------------------------------------------

    def describe(self) -> Dict[str, float]:
        """The headline numbers as a dict — handy for tables and JSON."""
        return {
            "trials": self.trials,
            "mean": self.mean,
            "std": self.std,
            "standard_error": self.standard_error,
            "min": float(self.values.min()),
            "median": self.median,
            "max": float(self.values.max()),
            "skewness": self.skewness,
            "var_95": self.var(0.95),
            "var_99": self.var(0.99),
            "var_995": self.var(0.995),
            "tvar_99": self.tvar(0.99),
            "tvar_995": self.tvar(0.995),
        }

    def summary(self, levels: Iterable[float] = (0.5, 0.75, 0.9, 0.95, 0.99, 0.995)) -> str:
        rows = [
            f"Monte Carlo — {self.label}",
            f"  trials       {self.trials:,}" + (f"   seed {self.seed}" if self.seed is not None else ""),
            f"  mean         {self.mean:,.2f}   ± {1.96 * self.standard_error:,.2f} (95% CI)",
            f"  std dev      {self.std:,.2f}   skew {self.skewness:+.2f}",
            f"  min / max    {float(self.values.min()):,.2f} / {float(self.values.max()):,.2f}",
            "",
            "  level        VaR              TVaR",
        ]
        for level in levels:
            rows.append(f"  {level:<12.3%} {self.var(level):>14,.2f}   {self.tvar(level):>14,.2f}")
        return "\n".join(rows)

    def to_frame(self):
        """A pandas DataFrame of inputs plus the output, if pandas is around."""
        try:
            import pandas  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("to_frame() needs pandas: pip install pandas") from exc
        data = dict(self.inputs)
        data[self.label] = self.values
        return pandas.DataFrame(data)

    def plot(self, kind: str = "distribution", **kwargs):
        """Shortcut into :mod:`riskpy.viz`.

        ``kind`` is any chart that takes a Result: ``distribution``,
        ``exceedance``, ``convergence``, ``tornado``, ``cdf``, ``density``,
        ``qq`` (pass ``dist=``) or ``dashboard``.
        """
        from . import viz

        plots = {
            "distribution": viz.distribution,
            "convergence": viz.convergence,
            "tornado": viz.tornado,
            "exceedance": viz.exceedance,
            "cdf": viz.cdf,
            "density": viz.density,
            "qq": viz.qq,
            "dashboard": viz.dashboard,
        }
        if kind not in plots:
            raise ValueError(f"kind must be one of {sorted(plots)}, got {kind!r}")
        return plots[kind](self, **kwargs)

    def __len__(self) -> int:
        return self.trials

    def __repr__(self) -> str:
        return (
            f"<Result {self.label}: {self.trials:,} trials, "
            f"mean {self.mean:,.2f}, p99.5 {self.var(0.995):,.2f}>"
        )


def _check_level(level: float) -> None:
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be a probability in (0, 1), e.g. 0.995 — got {level!r}")


def _rankdata(values, np):
    """Ranks with ties averaged, fully vectorised (no Python loop over values)."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inverse, weights=ranks)
        ranks = (sums / counts)[inverse]
    return ranks


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class Model:
    """A set of named random inputs, optional dependence, and a formula."""

    def __init__(self, **variables: Distribution):
        for name, dist in variables.items():
            if not isinstance(dist, Distribution):
                raise TypeError(
                    f"{name!r} must be a Distribution, got {type(dist).__name__}. "
                    f"Wrap a fixed number in Constant({dist!r})."
                )
        self.variables: Dict[str, Distribution] = dict(variables)
        self.correlations: Dict[Tuple[str, str], float] = {}
        self._formula: Optional[Callable[..., Any]] = None

    def formula(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register the formula. Usable as a decorator; returns ``fn`` unchanged."""
        self._formula = fn
        return fn

    def correlate(self, a: str, b: str, rho: float) -> "Model":
        """Declare a Spearman rank correlation between two inputs. Chainable."""
        for name in (a, b):
            if name not in self.variables:
                raise ValueError(f"unknown variable {name!r}; known: {list(self.variables)}")
        self.correlations[(a, b)] = float(rho)
        correlation_matrix(list(self.variables), self.correlations)  # validate now
        return self

    def sample(self, trials: int = 10_000, seed: Optional[int] = None, sampling: str = "random",
               copula: Any = None) -> Dict[str, Any]:
        """Draw the inputs only, as ``{name: array}`` — for inspection or for
        feeding into something that is not a formula."""
        np = _numpy()
        rng = np.random.default_rng(seed)
        return _draw(self.variables, trials, rng, sampling, self.correlations or None, copula)

    def run(
        self,
        trials: int = 10_000,
        seed: Optional[int] = None,
        fn: Optional[Callable[..., Any]] = None,
        vectorised: bool = True,
        keep_inputs: bool = True,
        label: Optional[str] = None,
        sampling: str = "random",
        copula: Any = None,
    ) -> Result:
        formula = fn or self._formula
        if formula is None:
            raise ValueError(
                "No formula registered. Decorate one with @model.formula, or "
                "pass fn= to run()."
            )
        return simulate(
            formula,
            trials=trials,
            seed=seed,
            vectorised=vectorised,
            keep_inputs=keep_inputs,
            label=label or getattr(formula, "__name__", "output"),
            correlation=self.correlations or None,
            sampling=sampling,
            copula=copula,
            **self.variables,
        )

    def sweep(self, name: str, alternatives: Mapping[str, Distribution], trials: int = 10_000,
              seed: Optional[int] = None, **run_kwargs) -> Dict[str, Result]:
        """Re-run the model with one input swapped for each alternative — the
        data for :func:`riskpy.viz.compare`. Same seed for every scenario, so
        the differences are the scenarios and not the noise."""
        if name not in self.variables:
            raise ValueError(f"unknown variable {name!r}; known: {list(self.variables)}")
        results = {}
        original = self.variables[name]
        try:
            for label, dist in alternatives.items():
                if not isinstance(dist, Distribution):
                    raise TypeError(f"{label!r}: alternatives must be Distributions")
                self.variables[name] = dist
                results[label] = self.run(trials=trials, seed=seed, label=label, **run_kwargs)
        finally:
            self.variables[name] = original
        return results

    def __repr__(self) -> str:
        names = ", ".join(self.variables)
        return f"<Model({names})>"


def _draw(variables: Mapping[str, Distribution], trials: int, rng, sampling: str,
          correlation, copula) -> Dict[str, Any]:
    np = _numpy()
    names = list(variables)
    if sampling not in ("random", "lhs"):
        raise ValueError(f"sampling must be 'random' or 'lhs', got {sampling!r}")

    if copula is not None:
        if correlation is not None:
            raise ValueError("pass either correlation= or copula=, not both")
        dim = getattr(copula, "dim", None)
        if dim is not None and dim != len(names):
            raise ValueError(f"copula has dimension {dim} but there are {len(names)} variables")
        u = np.asarray(copula.uniforms(trials, rng), dtype=float)
        if u.shape != (trials, len(names)):
            raise ValueError(f"copula.uniforms must return shape ({trials}, {len(names)}), got {u.shape}")
        return {name: variables[name].ppf(u[:, j]) for j, name in enumerate(names)}

    if sampling == "lhs":
        samples = {name: dist.sample_lhs(trials, rng) for name, dist in variables.items()}
    else:
        samples = {name: dist.sample(trials, rng) for name, dist in variables.items()}

    if correlation is not None:
        if isinstance(correlation, Mapping):
            matrix = correlation_matrix(names, correlation)
        else:
            matrix = np.asarray(correlation, dtype=float)
            if matrix.shape != (len(names), len(names)):
                raise ValueError(
                    f"correlation matrix must be {len(names)}x{len(names)} in the order "
                    f"{names}, got shape {matrix.shape}"
                )
        samples = iman_conover(samples, matrix, rng)
    return samples


def simulate(
    fn: Callable[..., Any],
    trials: int = 10_000,
    seed: Optional[int] = None,
    vectorised: bool = True,
    keep_inputs: bool = True,
    label: str = "output",
    correlation: Any = None,
    sampling: str = "random",
    copula: Any = None,
    **variables: Distribution,
) -> Result:
    """Run ``fn`` against ``trials`` draws of each named distribution.

    ``fn`` receives one keyword argument per variable. Vectorised (the default)
    means it is called once with arrays of length ``trials``; set
    ``vectorised=False`` if the formula contains branching that NumPy cannot
    express, and it will be called once per trial instead.

    ``correlation`` is ``{(a, b): rho}`` (Spearman) or a full matrix in the
    order the variables were given; it is imposed by Iman–Conover reordering,
    so the marginals are exactly what you asked for. ``sampling="lhs"`` uses
    Latin hypercube draws, which cuts the noise on the mean by a large factor
    at no cost. ``copula`` accepts any object with ``uniforms(n, rng)`` — see
    :mod:`riskpy.capital` — for tail dependence a correlation cannot express.
    """
    np = _numpy()

    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials!r}")
    if not variables:
        raise ValueError("Nothing to simulate — pass at least one distribution.")
    for name, dist in variables.items():
        if not isinstance(dist, Distribution):
            raise TypeError(
                f"{name!r} must be a Distribution, got {type(dist).__name__}. "
                f"Wrap a fixed number in Constant({dist!r})."
            )

    rng = np.random.default_rng(seed)
    samples = _draw(variables, trials, rng, sampling, correlation, copula)

    if vectorised:
        values = np.asarray(fn(**samples), dtype=float)
        if values.shape != (trials,):
            raise ValueError(
                f"The formula returned shape {values.shape}, expected ({trials},). "
                f"A vectorised formula must operate elementwise on its inputs — "
                f"use np.where instead of if/else, np.minimum instead of min(), "
                f"and so on. Or pass vectorised=False to loop instead."
            )
    else:
        values = np.empty(trials, dtype=float)
        for i in range(trials):
            values[i] = fn(**{name: column[i] for name, column in samples.items()})

    if not np.all(np.isfinite(values)):
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"{bad:,} of {trials:,} trials produced NaN or infinity. Usually a "
            f"division by a variable that can reach zero, or a log of a variable "
            f"that can go negative."
        )

    return Result(
        values=values,
        inputs=samples if keep_inputs else {},
        trials=trials,
        seed=seed,
        label=label,
    )
