"""Economic capital — dependence between risks, aggregation, and allocation.

A capital model answers three questions in a row, and this module has one piece for each:

**How do the risks move together?** A correlation says how often two risks are bad at once
on an ordinary day; it says nothing about whether they are bad together in the tail. A
*copula* does. Five families are here, each with its closed forms (Kendall's tau, tail
dependence, the bivariate CDF) next to its sampler:

    from riskpy.capital import GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula

    cop = StudentTCopula(0.5, df=4)
    cop.kendall_tau()            # 0.3333…  = 2/π · asin(0.5)
    cop.tail_dependence()        # TailDependence(lower=0.2532, upper=0.2532)
    cop.joint_exceedance(0.995)  # P(both beyond their 1-in-200)

    model.run(100_000, seed=1, copula=cop)        # any riskpy.mc Model takes one

**How much capital does the whole need?** :func:`aggregate_capital` is the
variance–covariance ("square-root") formula on standalone capital figures;
:func:`allocate` works on simulated losses and any risk measure.

**Whose capital is it?** Both return the Euler allocation — each line's contribution, which
adds up to the total exactly:

    from riskpy.capital import allocate, aggregate_capital

    agg = aggregate_capital({"market": 100, "credit": 60, "insurance": 80},
                            {("market", "credit"): 0.5, ("market", "insurance"): 0.25,
                             ("credit", "insurance"): 0.25})
    agg.total, agg.contributions, agg.diversification_benefit

    losses = model.sample(200_000, seed=1, copula=cop)      # {name: array}, one joint draw per trial
    alloc = allocate(losses, level=0.99)                    # co-TVaR (expected shortfall) contributions
    print(alloc.summary())

:func:`nearest_correlation` repairs a correlation matrix that is not positive
semi-definite (Higham's alternating projections), which is what expert-judgement matrices
usually are.

Conventions: losses are positive numbers (a bigger number is a worse outcome), ``level`` is
a decimal (0.995), correlations are decimals in [-1, 1]. A copula's ``correlation`` is the
correlation of its latent normal (or t) variables — not the Spearman rank correlation that
``Model.correlate`` takes; ``kendall_tau()`` and ``spearman_rho()`` convert. TVaR means
expected shortfall (Acerbi–Tasche), the same definition as ``Result.tvar``.

The closed forms are pure Python; sampling, :func:`allocate` and
:func:`nearest_correlation` need NumPy (``pip install "open-riskpy[sim]"``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from . import _special
from .errors import ArgumentTypeError, InputError

__all__ = [
    "Copula",
    "GaussianCopula",
    "StudentTCopula",
    "ClaytonCopula",
    "GumbelCopula",
    "FrankCopula",
    "TailDependence",
    "empirical_tail_dependence",
    "nearest_correlation",
    "Aggregation",
    "aggregate_capital",
    "Allocation",
    "allocate",
]


def _numpy():
    from .mc import _numpy as mc_numpy

    return mc_numpy()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_level(level: Any, name: str = "level") -> float:
    if not _is_number(level) or not 0.0 < float(level) < 1.0:
        raise InputError(f"{name} must be strictly between 0 and 1, got {level!r}", code="RP110", param=name,
                         hint="A confidence level is a decimal: 99.5% is 0.995.")
    return float(level)


# ---------------------------------------------------------------------------
# Closed forms (pure Python)


def _adaptive_simpson(f, a: float, b: float, tol: float = 1e-11, depth: int = 40) -> float:
    """∫ₐᵇ f by adaptive Simpson: intervals are halved until the Richardson error meets ``tol``."""
    def recurse(lo, flo, hi, fhi, mid, fmid, whole, tolerance, level):
        left_mid, right_mid = 0.5 * (lo + mid), 0.5 * (mid + hi)
        f_left, f_right = f(left_mid), f(right_mid)
        left = (mid - lo) / 6.0 * (flo + 4.0 * f_left + fmid)
        right = (hi - mid) / 6.0 * (fmid + 4.0 * f_right + fhi)
        delta = left + right - whole
        if level <= 0 or abs(delta) <= 15.0 * tolerance:
            return left + right + delta / 15.0
        return (recurse(lo, flo, mid, fmid, left_mid, f_left, left, 0.5 * tolerance, level - 1)
                + recurse(mid, fmid, hi, fhi, right_mid, f_right, right, 0.5 * tolerance, level - 1))

    if a == b:
        return 0.0
    total = 0.0
    pieces = 16                    # a fixed first split, so a narrow peak cannot hide between three points
    width = (b - a) / pieces
    for k in range(pieces):
        lo, hi = a + k * width, a + (k + 1) * width
        mid = 0.5 * (lo + hi)
        flo, fmid, fhi = f(lo), f(mid), f(hi)
        total += recurse(lo, flo, hi, fhi, mid, fmid, (hi - lo) / 6.0 * (flo + 4.0 * fmid + fhi), tol / pieces, depth)
    return total


def _plackett_integral(a: float, b: float, lower: float, upper: float, df: Optional[float]) -> float:
    """(1/2π)·∫ of exp(−q/2) (normal, df None) or (1 + q/df)^(−df/2) (Student t) over θ in
    [lower, upper], with q = (a² + b² − 2ab·sin θ)/cos²θ.

    For the normal this is Plackett's identity, ∂Φ₂(a, b; ρ)/∂ρ = the bivariate density,
    after substituting ρ = sin θ to remove the 1/√(1 − ρ²) singularity. The Student t
    version follows by mixing over the chi-square scale: E[exp(−W·q/2)] = (1 + q/df)^(−df/2).
    """
    if lower == upper:
        return 0.0
    plus, minus, ab = (a + b) ** 2, (a - b) ** 2, a * b

    def integrand(theta: float) -> float:
        sin_t = math.sin(theta)
        c2 = (1.0 - sin_t) * (1.0 + sin_t)
        # Two algebraically equal forms of q; each is stable at one end of (−π/2, π/2).
        if theta < 0.0:
            head, rest = plus, -2.0 * ab / (1.0 - sin_t)
        else:
            head, rest = minus, 2.0 * ab / (1.0 + sin_t)
        if head == 0.0:
            q = rest
        elif c2 <= 0.0:
            return 0.0
        else:
            q = head / c2 + rest
        q = max(q, 0.0)
        if df is None:
            return math.exp(-0.5 * q) if q < 1400.0 else 0.0
        return math.exp(-0.5 * df * math.log1p(q / df))

    return _adaptive_simpson(integrand, lower, upper) / (2.0 * math.pi)


def _bvn_cdf(a: float, b: float, rho: float) -> float:
    """P(X ≤ a, Y ≤ b) for a standard bivariate normal with correlation ``rho``."""
    pa, pb = _special.norm_cdf(a), _special.norm_cdf(b)
    if rho >= 1.0:
        return min(pa, pb)
    if rho <= -1.0:
        return max(pa + pb - 1.0, 0.0)
    # From independence (ρ = 0), where the joint probability is the product.
    return min(max(pa * pb + _plackett_integral(a, b, 0.0, math.asin(rho), None), 0.0), min(pa, pb))


def _bvt_cdf(a: float, b: float, rho: float, df: float) -> float:
    """P(X ≤ a, Y ≤ b) for a standard bivariate Student t with ``df`` degrees of freedom."""
    pa, pb = _special.t_cdf(a, df), _special.t_cdf(b, df)
    upper, lower = min(pa, pb), max(pa + pb - 1.0, 0.0)
    if rho >= 1.0:
        return upper
    if rho <= -1.0:
        return lower
    # An uncorrelated bivariate t is not independent, so there is no product to start from:
    # integrate from the nearer of the two perfectly dependent cases instead.
    if rho >= 0.0:
        value = upper - _plackett_integral(a, b, math.asin(rho), 0.5 * math.pi, df)
    else:
        value = lower + _plackett_integral(a, b, -0.5 * math.pi, math.asin(rho), df)
    return min(max(value, lower), upper)


def _debye1(x: float) -> float:
    """The first Debye function D₁(x) = (1/x)·∫₀ˣ t/(eᵗ − 1) dt, for x > 0."""
    if x < 1e-8:
        return 1.0 - x / 4.0

    def integrand(t: float) -> float:
        return 1.0 if t < 1e-12 else t / math.expm1(t)

    return _special.simpson(integrand, 0.0, x, n=2000) / x


def _frank_tau(theta: float) -> float:
    if abs(theta) < 1e-6:
        return theta / 9.0
    d1 = _debye1(abs(theta))
    if theta < 0.0:
        d1 += -theta / 2.0          # D₁(−x) = D₁(x) + x/2
    return 1.0 - 4.0 / theta * (1.0 - d1)


class TailDependence(NamedTuple):
    """The coefficients of lower and upper tail dependence, each in [0, 1].

    ``upper`` is the limit of P(V > q | U > q) as q → 1: the chance that one risk is beyond
    an extreme quantile given that the other is. Zero means the risks are asymptotically
    independent however strongly they are correlated in the body of the distribution.
    """

    lower: float
    upper: float


# ---------------------------------------------------------------------------
# Copulas


class Copula:
    """Base class for a copula: the dependence between risks, separated from their marginals.

    Every copula here has ``dim``, draws ``uniforms(n, rng)`` (an ``(n, dim)`` array whose
    columns are each uniform on (0, 1) and jointly dependent), and reports its closed-form
    ``kendall_tau()`` and ``tail_dependence()``. Pass one as ``copula=`` to
    :meth:`riskpy.mc.Model.run`, :meth:`riskpy.mc.Model.sample` or
    :func:`riskpy.mc.simulate`: the model's variables, in the order they were declared, take
    the columns.

    To write your own, subclass this and implement ``uniforms``.
    """

    dim: int = 2

    def uniforms(self, n: int, rng: Any):  # pragma: no cover - abstract
        """Draw ``n`` joint observations as an ``(n, dim)`` array of uniforms, using ``rng``
        (a ``numpy.random.Generator``)."""
        raise NotImplementedError

    def sample(self, n: int, seed: Optional[int] = None):
        """Draw ``n`` joint observations as an ``(n, dim)`` array of uniforms.

        n: number of draws
        seed: random seed; the same seed gives the same draws

        Example::

            u = ClaytonCopula(2.0).sample(10_000, seed=1)     # shape (10000, 2)
        """
        np = _numpy()
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise InputError(f"n must be a whole number >= 1, got {n!r}", code="RP107", param="n")
        return self.uniforms(int(n), np.random.default_rng(seed))

    def kendall_tau(self) -> float:  # pragma: no cover - abstract
        """Kendall's rank correlation implied by the copula's parameters."""
        raise NotImplementedError

    def tail_dependence(self) -> TailDependence:  # pragma: no cover - abstract
        """The coefficients of lower and upper tail dependence."""
        raise NotImplementedError

    def cdf(self, u: float, v: float) -> float:  # pragma: no cover - abstract
        """C(u, v) = P(U ≤ u, V ≤ v) for a pair of the copula's variables."""
        raise NotImplementedError

    def joint_exceedance(self, level: float) -> float:
        """P(U > level, V > level): the chance both risks are beyond the same quantile at once.

        level: the quantile as a decimal, e.g. 0.995 for the 1-in-200

        Independent risks give ``(1 − level)²`` (2.5e-5 at 0.995); comonotonic risks give
        ``1 − level``. Computed as ``1 − 2·level + C(level, level)``.

        Example::

            GumbelCopula(2.0).joint_exceedance(0.995)     # 0.00293, 117× the independent 2.5e-5
        """
        q = _check_level(level)
        return max(1.0 - 2.0 * q + self.cdf(q, q), 0.0)

    def _check_pair(self, u: Any, v: Any) -> Tuple[float, float]:
        for name, value in (("u", u), ("v", v)):
            if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
                raise InputError(f"{name} must be a probability in [0, 1], got {value!r}", code="RP103", param=name)
        return float(u), float(v)


def _tau_to_rho(tau: Any) -> float:
    if not _is_number(tau) or not -1.0 <= float(tau) <= 1.0:
        raise InputError(f"tau must be in [-1, 1], got {tau!r}", code="RP100", param="tau")
    return math.sin(0.5 * math.pi * float(tau))


class _Elliptical(Copula):
    """Shared machinery of the Gaussian and Student t copulas: a correlation matrix."""

    def __init__(self, correlation: Any):
        if _is_number(correlation):
            rho = float(correlation)
            if not -1.0 <= rho <= 1.0:
                raise InputError(f"correlation must be in [-1, 1], got {correlation!r}", code="RP100",
                                 param="correlation", hint="Correlations are decimals in [-1, 1].")
            self._matrix: List[List[float]] = [[1.0, rho], [rho, 1.0]]
        else:
            from .mc import _check_correlation

            np = _numpy()
            matrix = np.asarray(correlation, dtype=float)
            _check_correlation(matrix)
            if matrix.shape[0] < 2:
                raise InputError("a copula needs at least two variables", code="RP111", param="correlation")
            self._matrix = matrix.tolist()
        self.dim = len(self._matrix)

    @property
    def correlation(self):
        """The correlation matrix of the latent variables, as a NumPy array."""
        return _numpy().array(self._matrix, dtype=float)

    def _rho(self, i: int, j: int) -> float:
        for name, k in (("i", i), ("j", j)):
            if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < self.dim:
                raise InputError(f"{name} must be a variable index from 0 to {self.dim - 1}, got {k!r}",
                                 code="RP100", param=name)
        if i == j:
            raise InputError("i and j must be two different variables", code="RP113", param="j")
        return self._matrix[i][j]

    def _latent(self, n: int, rng: Any):
        """``n`` correlated standard normal rows. An eigen-decomposition rather than a
        Cholesky factor, so a semi-definite matrix (perfectly correlated risks) works."""
        np = _numpy()
        values, vectors = np.linalg.eigh(np.array(self._matrix, dtype=float))
        root = vectors * np.sqrt(np.clip(values, 0.0, None))
        return rng.standard_normal((n, self.dim)) @ root.T

    def kendall_tau(self, i: int = 0, j: int = 1) -> float:
        """Kendall's tau between variables ``i`` and ``j``: ``2/π · asin(ρ)``.

        The same formula holds for every elliptical copula, so a Gaussian and a t copula
        with the same correlation have the same tau and differ only in the tails.

        Example::

            GaussianCopula(0.5).kendall_tau()     # 0.33333…
        """
        return 2.0 / math.pi * math.asin(self._rho(i, j))

    def __repr__(self) -> str:
        if self.dim == 2:
            return f"{type(self).__name__}({self._matrix[0][1]!r}{self._extra_repr()})"
        return f"{type(self).__name__}(<{self.dim}x{self.dim} correlation>{self._extra_repr()})"

    def _extra_repr(self) -> str:
        return ""


class GaussianCopula(_Elliptical):
    """The dependence of a multivariate normal: correlation, and no tail dependence.

    correlation: a number in [-1, 1] for two risks, or a full correlation matrix (ones on the
        diagonal, symmetric, positive semi-definite) in the order of the model's variables.
        It is the correlation of the latent normals; ``kendall_tau()`` and
        ``spearman_rho()`` give the rank correlations it implies.

    However high the correlation (short of 1), joint extremes become independent in the
    limit: ``tail_dependence()`` is (0, 0). That is the property that made the Gaussian
    copula understate joint defaults; use :class:`StudentTCopula` or :class:`GumbelCopula`
    when risks crash together.

    Example::

        cop = GaussianCopula([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
        u = cop.sample(100_000, seed=1)            # (100000, 3) uniforms
        model.run(100_000, seed=1, copula=cop)     # a three-variable riskpy.mc Model
    """

    def uniforms(self, n: int, rng: Any):
        """Draw ``n`` joint observations: correlated normals pushed through Φ."""
        return _open_unit(_special.norm_cdf_vec(self._latent(n, rng)))

    def spearman_rho(self, i: int = 0, j: int = 1) -> float:
        """Spearman's rank correlation between variables ``i`` and ``j``: ``6/π · asin(ρ/2)``.

        This is the number ``Model.correlate`` takes, so it converts between the two ways
        of declaring dependence.

        Example::

            GaussianCopula(0.5).spearman_rho()     # 0.48261…
        """
        return 6.0 / math.pi * math.asin(0.5 * self._rho(i, j))

    def tail_dependence(self, i: int = 0, j: int = 1) -> TailDependence:
        """(0, 0) for any correlation below 1 — the Gaussian copula has no tail dependence."""
        lam = 1.0 if self._rho(i, j) >= 1.0 else 0.0
        return TailDependence(lam, lam)

    def cdf(self, u: float, v: float, i: int = 0, j: int = 1) -> float:
        """C(u, v) = Φ₂(Φ⁻¹(u), Φ⁻¹(v); ρ) for variables ``i`` and ``j``.

        Example::

            GaussianCopula(0.5).cdf(0.5, 0.5)     # 1/4 + asin(0.5)/(2π) = 0.33333…
        """
        u, v = self._check_pair(u, v)
        rho = self._rho(i, j)
        if u == 0.0 or v == 0.0:
            return 0.0
        if u == 1.0:
            return v
        if v == 1.0:
            return u
        return _bvn_cdf(_special.norm_ppf(u), _special.norm_ppf(v), rho)

    def joint_exceedance(self, level: float, i: int = 0, j: int = 1) -> float:
        """P(both variables ``i`` and ``j`` exceed their ``level`` quantile). See :meth:`Copula.joint_exceedance`."""
        q = _check_level(level)
        return max(1.0 - 2.0 * q + self.cdf(q, q, i, j), 0.0)

    @classmethod
    def from_kendall_tau(cls, tau: float) -> "GaussianCopula":
        """The bivariate Gaussian copula with Kendall's tau ``tau``: ``ρ = sin(π·τ/2)``.

        Example::

            GaussianCopula.from_kendall_tau(1 / 3)     # GaussianCopula(0.5)
        """
        return cls(_tau_to_rho(tau))


class StudentTCopula(_Elliptical):
    """The dependence of a multivariate Student t: correlation plus symmetric tail dependence.

    correlation: a number in [-1, 1] for two risks, or a full correlation matrix
    df: degrees of freedom, > 0. Small values (3–5) mean strong tail dependence; as ``df``
        grows the copula tends to the Gaussian.

    Unlike the Gaussian, a t copula has tail dependence even at zero correlation: the shared
    chi-square mixing variable makes every risk extreme in the same trials.

    Example::

        cop = StudentTCopula(0.5, df=4)
        cop.tail_dependence()        # TailDependence(lower=0.25317, upper=0.25317)
        model.run(100_000, seed=1, copula=cop)
    """

    def __init__(self, correlation: Any, df: float):
        super().__init__(correlation)
        if not _is_number(df) or not math.isfinite(df) or df <= 0.0:
            raise InputError(f"df must be a positive number, got {df!r}", code="RP102", param="df")
        self.df = float(df)

    def _extra_repr(self) -> str:
        return f", df={self.df:g}"

    def uniforms(self, n: int, rng: Any):
        """Draw ``n`` joint observations: correlated normals over a shared √(χ²/df), through the t CDF."""
        np = _numpy()
        z = self._latent(n, rng)
        w = rng.chisquare(self.df, size=n) / self.df
        nu = self.df
        # With very few degrees of freedom the mixing variable can underflow to zero, which
        # sends the whole row to ±infinity: the right limit, reached without a warning.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            t = z / np.sqrt(w)[:, None]
            x = np.nan_to_num(nu / (nu + t * t), nan=1.0, posinf=1.0, neginf=0.0)
        ib = _special.betainc_vec(0.5 * nu, 0.5, np.clip(x, 0.0, 1.0))
        return _open_unit(np.where(z > 0.0, 1.0 - 0.5 * ib, 0.5 * ib))

    def tail_dependence(self, i: int = 0, j: int = 1) -> TailDependence:
        """``λ = 2·t_{df+1}(−√((df + 1)(1 − ρ)/(1 + ρ)))``, the same in both tails.

        Example::

            StudentTCopula(0.5, df=4).tail_dependence().upper     # 0.25317
        """
        rho = self._rho(i, j)
        if rho >= 1.0:
            return TailDependence(1.0, 1.0)
        if rho <= -1.0:
            return TailDependence(0.0, 0.0)
        lam = 2.0 * _special.t_cdf(-math.sqrt((self.df + 1.0) * (1.0 - rho) / (1.0 + rho)), self.df + 1.0)
        return TailDependence(lam, lam)

    def cdf(self, u: float, v: float, i: int = 0, j: int = 1) -> float:
        """C(u, v) = T₂(t⁻¹(u), t⁻¹(v); ρ, df) for variables ``i`` and ``j``.

        Example::

            StudentTCopula(0.5, df=4).cdf(0.5, 0.5)     # 0.33333…, as for any elliptical copula
        """
        u, v = self._check_pair(u, v)
        rho = self._rho(i, j)
        if u == 0.0 or v == 0.0:
            return 0.0
        if u == 1.0:
            return v
        if v == 1.0:
            return u
        try:
            a, b = _special.t_ppf(u, self.df), _special.t_ppf(v, self.df)
        except OverflowError:
            a = b = math.inf
        if not (abs(a) < 1e150 and abs(b) < 1e150):
            raise InputError(f"the t quantile at this level is beyond double precision for df = {self.df:g}",
                             code="RP304", param="df",
                             hint="Use more degrees of freedom (df >= 1 is safe at any level) or a level nearer 0.5.")
        return _bvt_cdf(a, b, rho, self.df)

    def joint_exceedance(self, level: float, i: int = 0, j: int = 1) -> float:
        """P(both variables ``i`` and ``j`` exceed their ``level`` quantile). See :meth:`Copula.joint_exceedance`."""
        q = _check_level(level)
        return max(1.0 - 2.0 * q + self.cdf(q, q, i, j), 0.0)

    @classmethod
    def from_kendall_tau(cls, tau: float, df: float) -> "StudentTCopula":
        """The bivariate t copula with Kendall's tau ``tau`` and ``df`` degrees of freedom.

        Example::

            StudentTCopula.from_kendall_tau(1 / 3, df=4)     # StudentTCopula(0.5, df=4)
        """
        return cls(_tau_to_rho(tau), df)


class _Archimedean(Copula):
    """Shared machinery of the one-parameter Archimedean copulas (exchangeable in ``dim``)."""

    _name = "theta"

    def __init__(self, theta: float, dim: int = 2, survival: bool = False):
        if not _is_number(theta) or not math.isfinite(theta):
            raise InputError(f"theta must be a finite number, got {theta!r}", code="RP101", param="theta")
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 2:
            raise InputError(f"dim must be a whole number >= 2, got {dim!r}", code="RP107", param="dim")
        if not isinstance(survival, bool):
            raise ArgumentTypeError(f"survival must be True or False, got {survival!r}", code="RP404", param="survival")
        self.theta = float(theta)
        self.dim = dim
        self.survival = survival

    def _draw(self, n: int, rng: Any):  # pragma: no cover - abstract
        raise NotImplementedError

    def uniforms(self, n: int, rng: Any):
        """Draw ``n`` joint observations as an ``(n, dim)`` array of uniforms."""
        u = _open_unit(self._draw(n, rng))
        return 1.0 - u if self.survival else u

    def _cdf(self, u: float, v: float) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def cdf(self, u: float, v: float) -> float:
        """C(u, v) = P(U ≤ u, V ≤ v) for any pair of the copula's variables (closed form)."""
        u, v = self._check_pair(u, v)
        if self.survival:
            # The survival copula of C: Ĉ(u, v) = u + v − 1 + C(1 − u, 1 − v).
            return min(max(u + v - 1.0 + self._edge(1.0 - u, 1.0 - v), 0.0), min(u, v))
        return self._edge(u, v)

    def _edge(self, u: float, v: float) -> float:
        if u == 0.0 or v == 0.0:
            return 0.0
        if u == 1.0:
            return v
        if v == 1.0:
            return u
        return min(max(self._cdf(u, v), 0.0), min(u, v))

    def _tails(self) -> TailDependence:  # pragma: no cover - abstract
        raise NotImplementedError

    def tail_dependence(self) -> TailDependence:
        """The coefficients of lower and upper tail dependence (swapped for ``survival=True``)."""
        tails = self._tails()
        return TailDependence(tails.upper, tails.lower) if self.survival else tails

    def __repr__(self) -> str:
        extra = (f", dim={self.dim}" if self.dim != 2 else "") + (", survival=True" if self.survival else "")
        return f"{type(self).__name__}({self.theta!r}{extra})"


class ClaytonCopula(_Archimedean):
    """Clayton copula: risks that are bad together at the *low* end (lower tail dependence).

    theta: dependence parameter, > 0. Kendall's tau is ``θ/(θ + 2)``.
    dim: number of risks (every pair has the same dependence)
    survival: True flips it, so the dependence is in the *upper* tail — the usual choice
        for losses, where the big outcomes are the ones that come together.

    ``C(u, v) = (u^−θ + v^−θ − 1)^(−1/θ)``; lower tail dependence ``2^(−1/θ)``, upper 0.

    Example::

        cop = ClaytonCopula(2.0, survival=True)       # joint large losses
        cop.kendall_tau()                             # 0.5
        cop.tail_dependence()                         # TailDependence(lower=0.0, upper=0.70711)
    """

    def __init__(self, theta: float, dim: int = 2, survival: bool = False):
        super().__init__(theta, dim, survival)
        if self.theta <= 0.0:
            raise InputError(f"theta must be > 0 for a Clayton copula, got {theta!r}", code="RP102", param="theta",
                             hint="theta = 2·tau/(1 − tau) for a Kendall's tau in (0, 1).")

    def _draw(self, n: int, rng: Any):
        # Marshall–Olkin: a Gamma(1/θ) frailty shared by the row, then ψ(E/V) with ψ(t) = (1 + t)^(−1/θ).
        v = rng.gamma(1.0 / self.theta, 1.0, size=n)[:, None]
        e = rng.exponential(1.0, size=(n, self.dim))
        np = _numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v > 0.0, (1.0 + e / np.where(v > 0.0, v, 1.0)) ** (-1.0 / self.theta), 0.0)

    def _cdf(self, u: float, v: float) -> float:
        # (u^−θ + v^−θ − 1)^(−1/θ), with the largest power factored out so a large θ cannot overflow.
        t = self.theta
        a, b = -t * math.log(u), -t * math.log(v)
        m = max(a, b)
        if m < 30.0:
            return math.exp(-math.log1p(math.expm1(a) + math.expm1(b)) / t)      # exact as θ → 0
        return math.exp(-(m + math.log(math.exp(a - m) + math.exp(b - m) - math.exp(-m))) / t)

    def kendall_tau(self) -> float:
        """Kendall's tau: ``θ/(θ + 2)``."""
        return self.theta / (self.theta + 2.0)

    def _tails(self) -> TailDependence:
        return TailDependence(2.0 ** (-1.0 / self.theta), 0.0)

    @classmethod
    def from_kendall_tau(cls, tau: float, dim: int = 2, survival: bool = False) -> "ClaytonCopula":
        """The Clayton copula with Kendall's tau ``tau`` in (0, 1): ``θ = 2τ/(1 − τ)``.

        Example::

            ClaytonCopula.from_kendall_tau(0.5)     # ClaytonCopula(2.0)
        """
        if not _is_number(tau) or not 0.0 < float(tau) < 1.0:
            raise InputError(f"tau must be strictly between 0 and 1 for a Clayton copula, got {tau!r}",
                             code="RP100", param="tau")
        return cls(2.0 * tau / (1.0 - tau), dim, survival)


class GumbelCopula(_Archimedean):
    """Gumbel copula: risks that are bad together at the *high* end (upper tail dependence).

    theta: dependence parameter, >= 1 (1 is independence). Kendall's tau is ``1 − 1/θ``.
    dim: number of risks (every pair has the same dependence)
    survival: True flips it, so the dependence is in the lower tail

    ``C(u, v) = exp(−((−ln u)^θ + (−ln v)^θ)^(1/θ))``; upper tail dependence ``2 − 2^(1/θ)``,
    lower 0. The natural copula for insurance losses, where one event drives several lines.

    Example::

        cop = GumbelCopula(2.0)
        cop.kendall_tau()            # 0.5
        cop.tail_dependence()        # TailDependence(lower=0.0, upper=0.58579)
    """

    def __init__(self, theta: float, dim: int = 2, survival: bool = False):
        super().__init__(theta, dim, survival)
        if self.theta < 1.0:
            raise InputError(f"theta must be >= 1 for a Gumbel copula, got {theta!r}", code="RP100", param="theta",
                             hint="theta = 1/(1 − tau) for a Kendall's tau in [0, 1); theta = 1 is independence.")

    def _draw(self, n: int, rng: Any):
        np = _numpy()
        e = rng.exponential(1.0, size=(n, self.dim))
        if self.theta == 1.0:
            return np.exp(-e)
        # Marshall–Olkin with a positive stable frailty of index α = 1/θ, drawn by Kanter's
        # (Chambers–Mallows–Stuck) formula; then ψ(E/V) with ψ(t) = exp(−t^α).
        alpha = 1.0 / self.theta
        angle = rng.uniform(0.0, math.pi, size=n)
        w = rng.exponential(1.0, size=n)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_v = (np.log(np.sin(alpha * angle)) - np.log(np.sin(angle)) / alpha
                     + (1.0 - alpha) / alpha * (np.log(np.sin((1.0 - alpha) * angle)) - np.log(w)))
            u = np.exp(-np.exp(alpha * (np.log(e) - log_v[:, None])))
        return np.nan_to_num(u, nan=0.5)

    def _cdf(self, u: float, v: float) -> float:
        t = self.theta
        return math.exp(-(((-math.log(u)) ** t + (-math.log(v)) ** t) ** (1.0 / t)))

    def kendall_tau(self) -> float:
        """Kendall's tau: ``1 − 1/θ``."""
        return 1.0 - 1.0 / self.theta

    def _tails(self) -> TailDependence:
        return TailDependence(0.0, 2.0 - 2.0 ** (1.0 / self.theta))

    @classmethod
    def from_kendall_tau(cls, tau: float, dim: int = 2, survival: bool = False) -> "GumbelCopula":
        """The Gumbel copula with Kendall's tau ``tau`` in [0, 1): ``θ = 1/(1 − τ)``.

        Example::

            GumbelCopula.from_kendall_tau(0.5)     # GumbelCopula(2.0)
        """
        if not _is_number(tau) or not 0.0 <= float(tau) < 1.0:
            raise InputError(f"tau must be in [0, 1) for a Gumbel copula, got {tau!r}", code="RP100", param="tau")
        return cls(1.0 / (1.0 - tau), dim, survival)


class FrankCopula(_Archimedean):
    """Frank copula: symmetric dependence with no tail dependence at all.

    theta: dependence parameter, any non-zero number; negative values give negative
        dependence (two risks only). Kendall's tau is ``1 − 4/θ·(1 − D₁(θ))`` with D₁ the
        first Debye function.
    dim: number of risks; more than two needs ``theta > 0``

    ``C(u, v) = −(1/θ)·ln(1 + (e^(−θu) − 1)(e^(−θv) − 1)/(e^(−θ) − 1))``. The comparison
    case: the same rank correlation as a Gumbel or Clayton, without the joint extremes.

    Example::

        cop = FrankCopula(5.736)
        cop.kendall_tau()            # 0.5000
        cop.tail_dependence()        # TailDependence(lower=0.0, upper=0.0)
    """

    def __init__(self, theta: float, dim: int = 2):
        super().__init__(theta, dim, False)
        if self.theta == 0.0:
            raise InputError("theta must be non-zero for a Frank copula (0 is independence)", code="RP100",
                             param="theta")
        if abs(self.theta) > 500.0:
            raise InputError(f"theta must be within ±500 for a Frank copula, got {theta!r}", code="RP100",
                             param="theta", hint="|theta| above about 40 is already almost perfect dependence.")
        if self.dim > 2 and self.theta < 0.0:
            raise InputError("a Frank copula in more than two dimensions needs theta > 0", code="RP113",
                             param="theta", hint="Three or more risks cannot all be negatively dependent this way.")

    def _draw(self, n: int, rng: Any):
        np = _numpy()
        t = self.theta
        if self.dim == 2:
            # Conditional inversion, valid for either sign of theta.
            u = rng.random(n)
            w = rng.random(n)
            a = np.exp(-t * u)
            v = -np.log1p(w * np.expm1(-t) / (w + (1.0 - w) * a)) / t
            return np.column_stack([u, v])
        # Marshall–Olkin with a logarithmic-series frailty; ψ(s) = −ln(1 − (1 − e^(−θ))·e^(−s))/θ.
        p = min(-math.expm1(-t), 1.0 - 2.0 ** -53)
        frailty = rng.logseries(p, size=n).astype(float)[:, None]
        e = rng.exponential(1.0, size=(n, self.dim))
        return -np.log1p(-p * np.exp(-e / frailty)) / t

    def _cdf(self, u: float, v: float) -> float:
        t = self.theta
        if t < 0.0:
            # Negative dependence is the positive case reflected: C_θ(u, v) = u − C_{−θ}(u, 1 − v).
            return u - _frank_cdf(u, 1.0 - v, -t)
        return _frank_cdf(u, v, t)

    def kendall_tau(self) -> float:
        """Kendall's tau: ``1 − 4/θ·(1 − D₁(θ))``."""
        return _frank_tau(self.theta)

    def _tails(self) -> TailDependence:
        return TailDependence(0.0, 0.0)

    @classmethod
    def from_kendall_tau(cls, tau: float, dim: int = 2) -> "FrankCopula":
        """The Frank copula with Kendall's tau ``tau`` (non-zero, in (−1, 1)), solved numerically.

        Example::

            FrankCopula.from_kendall_tau(0.5).theta     # 5.7363
        """
        if not _is_number(tau) or not -1.0 < float(tau) < 1.0 or tau == 0.0:
            raise InputError(f"tau must be non-zero and strictly between -1 and 1, got {tau!r}", code="RP100",
                             param="tau")
        if abs(tau) > 0.99:
            raise InputError(f"|tau| must be at most 0.99 for a Frank copula, got {tau!r}", code="RP100", param="tau")
        lo, hi = (1e-6, 500.0) if tau > 0 else (-500.0, -1e-6)
        return cls(_special.brent(lambda th: _frank_tau(th) - tau, lo, hi, tol=1e-12), dim)


def _frank_cdf(u: float, v: float, t: float) -> float:
    """The Frank copula C(u, v) for θ = t > 0, arranged so that a large θ neither overflows nor cancels.

    −(1/θ)·ln(1 + (e^(−θu) − 1)(e^(−θv) − 1)/(e^(−θ) − 1)) with the bracket written as
    (e^(−θu) + e^(−θv) − e^(−θ) − e^(−θ(u+v))) / (1 − e^(−θ)) and e^(−θ·min(u, v)) factored out.
    """
    if t < 1.0:
        return -math.log1p(math.expm1(-t * u) * math.expm1(-t * v) / math.expm1(-t)) / t
    lo, hi = min(u, v), max(u, v)
    inner = 1.0 + math.exp(-t * (hi - lo)) - math.exp(-t * (1.0 - lo)) - math.exp(-t * hi)
    if inner <= 0.0:
        return 0.0
    return lo - (math.log(inner) - math.log(-math.expm1(-t))) / t


def _open_unit(u):
    """Keep simulated uniforms strictly inside (0, 1), so no quantile function is asked for ±infinity."""
    return _numpy().clip(u, 1e-16, 1.0 - 1e-16)


def empirical_tail_dependence(x: Any, y: Any, level: float = 0.95) -> TailDependence:
    """Estimate tail dependence from paired observations, at a finite quantile.

    x: observations of the first risk
    y: observations of the second, paired with ``x``
    level: the quantile defining "the tail", a decimal in (0.5, 1); the lower tail uses ``1 − level``

    Returns ``TailDependence(lower, upper)`` where ``upper = P(Y > its level-quantile | X >
    its level-quantile)``, counted on ranks. This is the finite-level conditional
    probability, which tends to the copula's coefficient only as ``level → 1``; at 0.95 a
    Gaussian copula with ρ = 0.5 still shows about 0.24, so compare it with
    ``copula.joint_exceedance(level) / (1 − level)``, not with ``tail_dependence()``.

    Example::

        u = GumbelCopula(2.0).sample(200_000, seed=1)
        empirical_tail_dependence(u[:, 0], u[:, 1], level=0.99).upper     # ≈ 0.59
    """
    np = _numpy()
    q = _check_level(level)
    if q <= 0.5:
        raise InputError(f"level must be above 0.5, got {level!r}", code="RP110", param="level")
    a, b = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if a.size != b.size:
        raise InputError(f"x and y must be paired: {a.size} and {b.size} observations", code="RP109", param="y")
    if a.size < 2:
        raise InputError("at least two paired observations are needed", code="RP111", param="x")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise InputError("x and y must be finite", code="RP101", param="x")
    n = a.size
    ra = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort")
    rb = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort")
    k = max(int(round(n * (1.0 - q))), 1)
    upper = float(np.sum((ra >= n - k) & (rb >= n - k))) / k
    lower = float(np.sum((ra < k) & (rb < k))) / k
    return TailDependence(lower, upper)


# ---------------------------------------------------------------------------
# Correlation matrices


def nearest_correlation(matrix: Any, tol: float = 1e-10, max_iter: int = 2000):
    """The nearest valid correlation matrix to ``matrix``, in the Frobenius norm.

    matrix: a square symmetric matrix of intended correlations, entries in [-1, 1]
    tol: relative change at which the iteration stops
    max_iter: iteration cap

    Returns a NumPy array: symmetric, ones on the diagonal, positive semi-definite. A matrix
    assembled from expert judgement or from pairwise estimates is often not positive
    semi-definite — the pairwise numbers cannot all hold at once — and then nothing can be
    simulated from it. This is Higham's (2002) alternating-projections algorithm with
    Dykstra's correction, which finds the closest matrix that can. A matrix that is already
    valid comes back unchanged.

    Example::

        nearest_correlation([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        # [[1.     0.7607 0.1573]
        #  [0.7607 1.     0.7607]
        #  [0.1573 0.7607 1.    ]]
    """
    np = _numpy()
    try:
        a = np.array(matrix, dtype=float)
    except (TypeError, ValueError):
        raise InputError("matrix must be a square array of numbers", code="RP100", param="matrix") from None
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
        raise InputError(f"matrix must be square, got shape {a.shape}", code="RP100", param="matrix")
    if not np.all(np.isfinite(a)):
        raise InputError("matrix must be finite", code="RP101", param="matrix")
    if not np.allclose(a, a.T, atol=1e-10):
        raise InputError("matrix must be symmetric", code="RP100", param="matrix")
    if np.abs(a).max() > 1.0 + 1e-12:
        raise InputError("correlations must be in [-1, 1]", code="RP100", param="matrix",
                         hint="Correlations are decimals in [-1, 1].")
    if not (_is_number(tol) and tol > 0.0):
        raise InputError(f"tol must be positive, got {tol!r}", code="RP102", param="tol")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise InputError(f"max_iter must be a whole number >= 1, got {max_iter!r}", code="RP107", param="max_iter")

    a = 0.5 * (a + a.T)
    target = a.copy()
    np.fill_diagonal(target, 1.0)
    if np.linalg.eigvalsh(target).min() >= -1e-12:
        return target

    y = a.copy()
    correction = np.zeros_like(a)
    for _ in range(max_iter):
        r = y - correction
        values, vectors = np.linalg.eigh(r)
        x = (vectors * np.clip(values, 0.0, None)) @ vectors.T
        correction = x - r
        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)
        change = np.linalg.norm(y_new - y, "fro") / max(np.linalg.norm(y_new, "fro"), 1e-300)
        y = y_new
        if change < tol:
            break
    # Finish on the positive semi-definite side: clip the last rounding-level negative
    # eigenvalue and rescale to a unit diagonal.
    values, vectors = np.linalg.eigh(0.5 * (y + y.T))
    y = (vectors * np.clip(values, 0.0, None)) @ vectors.T
    d = np.sqrt(np.clip(np.diag(y), 1e-300, None))
    y = y / np.outer(d, d)
    y = np.clip(0.5 * (y + y.T), -1.0, 1.0)
    np.fill_diagonal(y, 1.0)
    return y


def _plain_matrix(correlation: Any, names: Sequence[str]) -> List[List[float]]:
    """A correlation given as nested lists, an array or ``{(a, b): rho}``, as validated lists (pure Python)."""
    k = len(names)
    if isinstance(correlation, Mapping):
        index = {name: i for i, name in enumerate(names)}
        matrix = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
        for pair, rho in correlation.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise InputError(f"correlation keys must be (name, name) pairs, got {pair!r}", code="RP100",
                                 param="correlation")
            a, b = pair
            for name in (a, b):
                if name not in index:
                    raise InputError(f"correlation refers to unknown risk {name!r}; known: {list(names)}",
                                     code="RP108", param="correlation")
            if a == b:
                raise InputError(f"cannot set a risk's correlation with itself ({a!r})", code="RP113",
                                 param="correlation")
            if not _is_number(rho):
                raise ArgumentTypeError(f"correlation for {pair!r} must be a number, got {rho!r}", code="RP404",
                                        param="correlation")
            matrix[index[a]][index[b]] = matrix[index[b]][index[a]] = float(rho)
    else:
        tolist = getattr(correlation, "tolist", None)
        rows = tolist() if callable(tolist) else correlation
        if not isinstance(rows, (list, tuple)) or len(rows) != k or any(
                not isinstance(row, (list, tuple)) or len(row) != k for row in rows):
            raise InputError(f"correlation must be a {k}x{k} matrix in the order {list(names)}", code="RP109",
                             param="correlation")
        for row in rows:
            for value in row:
                if not _is_number(value):
                    raise ArgumentTypeError(f"correlation entries must be numbers, got {value!r}", code="RP404",
                                            param="correlation")
        matrix = [[float(v) for v in row] for row in rows]
    for i in range(k):
        if not all(math.isfinite(v) for v in matrix[i]):
            raise InputError("correlation must be finite", code="RP101", param="correlation")
        if abs(matrix[i][i] - 1.0) > 1e-12:
            raise InputError("correlation matrix must have ones on the diagonal", code="RP100", param="correlation")
        for j in range(k):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-12:
                raise InputError("correlation matrix must be symmetric", code="RP100", param="correlation")
            if not -1.0 <= matrix[i][j] <= 1.0:
                raise InputError(f"correlation must be in [-1, 1], got {matrix[i][j]!r}", code="RP100",
                                 param="correlation", hint="Correlations are decimals in [-1, 1].")
    if not _is_psd(matrix):
        raise InputError("correlation matrix is not positive semi-definite: the pairwise correlations cannot all "
                         "hold at once", code="RP100", param="correlation",
                         hint="Reduce the strongest correlations, or repair the matrix with "
                              "riskpy.capital.nearest_correlation() (the nearest_correlation tool).")
    return matrix


def _is_psd(matrix: Sequence[Sequence[float]], tol: float = 1e-10) -> bool:
    """Positive semi-definiteness by a diagonally pivoted Cholesky factorisation (pure Python)."""
    k = len(matrix)
    a = [list(row) for row in matrix]
    for step in range(k):
        pivot = max(range(step, k), key=lambda i: a[i][i])
        if a[pivot][pivot] < -tol:
            return False
        if a[pivot][pivot] <= tol:
            # Every remaining diagonal is ~0, so the remaining block must vanish.
            return all(abs(a[i][j]) <= 1e-7 for i in range(step, k) for j in range(step, k))
        if pivot != step:
            a[step], a[pivot] = a[pivot], a[step]
            for row in a:
                row[step], row[pivot] = row[pivot], row[step]
        d = a[step][step]
        for i in range(step + 1, k):
            f = a[i][step] / d
            for j in range(step + 1, k):
                a[i][j] -= f * a[step][j]
    return True


# ---------------------------------------------------------------------------
# Aggregation: the variance–covariance formula


def _fmt_money(value: float) -> str:
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude >= 1e15 or magnitude < 1e-3):
        return f"{value:.6g}"
    if magnitude >= 1000.0:
        return f"{value:,.0f}"
    return f"{value:,.4f}"


def _text_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(str(headers[c])), *(len(str(r[c])) for r in rows)) for c in range(len(headers))]

    def line(cells: Sequence[str]) -> str:
        first = str(cells[0]).ljust(widths[0])
        rest = [str(c).rjust(w) for c, w in zip(cells[1:], widths[1:])]
        return "  ".join([first, *rest]).rstrip()

    return "\n".join([line(headers), line(["-" * w for w in widths]), *(line(r) for r in rows)])


@dataclass(frozen=True)
class Aggregation:
    """Diversified capital from standalone figures, with each risk's Euler contribution.

    Fields: ``names``; ``standalone`` (``{name: capital}``, as given); ``correlation`` (the
    matrix used, nested tuples); ``total`` (the diversified capital, ``√(c′Rc)``);
    ``contributions`` (``{name: cᵢ·(Rc)ᵢ/total}``, which add up to ``total``).
    """

    names: Tuple[str, ...]
    standalone: Dict[str, float]
    correlation: Tuple[Tuple[float, ...], ...]
    total: float
    contributions: Dict[str, float]

    @property
    def undiversified(self) -> float:
        """The sum of the standalone figures — the capital with no diversification credit."""
        return math.fsum(self.standalone.values())

    @property
    def diversification_benefit(self) -> float:
        """``undiversified − total``, in the units of the capital figures."""
        return self.undiversified - self.total

    @property
    def diversification_ratio(self) -> float:
        """``1 − total/undiversified``: the share of the simple sum that diversification removes."""
        return 1.0 - self.total / self.undiversified if self.undiversified else 0.0

    @property
    def shares(self) -> Dict[str, float]:
        """Each risk's contribution as a fraction of the total (decimals that add up to 1)."""
        return {n: (c / self.total if self.total else 0.0) for n, c in self.contributions.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Everything as plain JSON-able data."""
        return {
            "total": self.total, "undiversified": self.undiversified,
            "diversification_benefit": self.diversification_benefit,
            "diversification_ratio": self.diversification_ratio,
            "standalone": dict(self.standalone), "contributions": dict(self.contributions),
            "shares": self.shares,
        }

    def summary(self) -> str:
        """A text table: standalone, contribution and share per risk, then the totals."""
        shares = self.shares
        rows = [[n, _fmt_money(self.standalone[n]), _fmt_money(self.contributions[n]), f"{shares[n]:.1%}"]
                for n in self.names]
        rows.append(["Total", _fmt_money(self.undiversified), _fmt_money(self.total), "100.0%"])
        return (_text_table(["Risk", "Standalone", "Contribution", "Share"], rows)
                + f"\nDiversification benefit {_fmt_money(self.diversification_benefit)} "
                  f"({self.diversification_ratio:.1%} of the standalone sum)")

    def plot(self, **kwargs):
        """The contributions as a bar chart (:func:`riskpy.viz.allocation`); needs Matplotlib.

        Keyword arguments (``title``, ``figsize``) go to the chart. Returns the Figure.

        Example::

            agg.plot(title="Diversified capital by risk").savefig("capital.png")
        """
        from . import viz

        return viz.allocation(self.contributions, total=self.total, **kwargs)

    def __repr__(self) -> str:
        return (f"<Aggregation total={self.total:.6g} undiversified={self.undiversified:.6g} "
                f"risks={len(self.names)}>")


def aggregate_capital(standalone: Any, correlation: Any, names: Optional[Sequence[str]] = None) -> Aggregation:
    """Combine standalone capital figures with a correlation matrix: ``total = √(c′Rc)``.

    standalone: the capital of each risk on its own — ``{name: amount}`` or a list of amounts,
        all >= 0, all at the same confidence level and in the same currency
    correlation: a matrix in the order of the risks, or ``{(name, name): rho}`` for the pairs
        that are correlated (unlisted pairs are independent)
    names: labels for the risks when ``standalone`` is a list (default ``risk_1, risk_2, …``)

    Returns an :class:`Aggregation`: the diversified ``total``, the ``undiversified`` sum,
    the ``diversification_benefit``, and each risk's Euler ``contributions``
    ``cᵢ·(Rc)ᵢ/total``, which add up to the total exactly.

    This is the variance–covariance formula used by the Solvency II standard formula and by
    most internal-model aggregation layers. It is exact when the risks are jointly elliptical
    (multivariate normal, for one) and each standalone figure is the same multiple of a
    standard deviation; for skewed risks or tail dependence it is an approximation, and
    :func:`allocate` on simulated losses is the tool. Pure Python: no NumPy needed.

    Example::

        agg = aggregate_capital({"market": 100.0, "credit": 60.0}, {("market", "credit"): 0.25})
        agg.total                      # 128.84 = √(100² + 60² + 2·0.25·100·60)
        agg.contributions              # {'market': 89.26, 'credit': 39.58}
        agg.diversification_benefit    # 31.16
    """
    if isinstance(standalone, Mapping):
        if names is not None:
            raise InputError("pass names only when standalone is a list", code="RP113", param="names")
        labels = [str(k) for k in standalone]
        amounts = list(standalone.values())
    else:
        tolist = getattr(standalone, "tolist", None)
        amounts = tolist() if callable(tolist) else standalone
        if not isinstance(amounts, (list, tuple)):
            raise ArgumentTypeError("standalone must be a list of amounts or a {name: amount} mapping, got "
                                    f"{type(standalone).__name__}", code="RP404", param="standalone")
        amounts = list(amounts)
        if names is None:
            labels = [f"risk_{i + 1}" for i in range(len(amounts))]
        else:
            labels = [str(n) for n in names]
            if len(labels) != len(amounts):
                raise InputError(f"names has {len(labels)} entries but standalone has {len(amounts)}",
                                 code="RP109", param="names")
    if not amounts:
        raise InputError("standalone is empty: give at least one capital figure", code="RP111", param="standalone")
    if len(set(labels)) != len(labels):
        raise InputError("risk names must be distinct", code="RP113", param="names")
    for label, amount in zip(labels, amounts):
        if not _is_number(amount):
            raise ArgumentTypeError(f"standalone capital for {label!r} must be a number, got {amount!r}",
                                    code="RP404", param="standalone")
        if not math.isfinite(amount):
            raise InputError(f"standalone capital for {label!r} must be finite", code="RP101", param="standalone")
        if amount < 0.0:
            raise InputError(f"standalone capital for {label!r} must be >= 0, got {amount!r}", code="RP102",
                             param="standalone", hint="Capital figures are positive amounts; a hedge is expressed "
                                                      "through a negative correlation, not a negative amount.")
    c = [float(a) for a in amounts]
    matrix = _plain_matrix(correlation, labels)
    # Work on figures scaled by the largest, so squaring them cannot overflow.
    scale = max(c)
    unit = [a / scale for a in c] if scale > 0.0 else c
    rc = [scale * math.fsum(matrix[i][j] * unit[j] for j in range(len(c))) for i in range(len(c))]
    total = scale * math.sqrt(max(math.fsum(unit[i] * rc[i] / scale for i in range(len(c))), 0.0)) \
        if scale > 0.0 else 0.0
    if not math.isfinite(total) or not math.isfinite(math.fsum(unit) * scale):
        raise InputError("the capital figures overflow double precision", code="RP304", param="standalone",
                         hint="Express the amounts in larger units (thousands or millions).")
    contributions = {label: (c[i] * (rc[i] / total) if total > 0.0 else 0.0) for i, label in enumerate(labels)}
    return Aggregation(tuple(labels), dict(zip(labels, c)), tuple(tuple(row) for row in matrix), total, contributions)


# ---------------------------------------------------------------------------
# Allocation from simulated losses


@dataclass(frozen=True)
class Allocation:
    """Capital for a portfolio of simulated losses, split between its lines.

    Fields: ``names``; ``level``; ``measure`` ("tvar" or "var"); ``method`` ("euler",
    "covariance" or "standalone"); ``trials``; ``total`` (the risk measure of the summed
    losses); ``standalone`` (``{name: risk measure of that line alone}``);
    ``contributions`` (``{name: allocated amount}``, adding up to ``total``); ``means``
    (``{name: expected loss}``).
    """

    names: Tuple[str, ...]
    level: float
    measure: str
    method: str
    trials: int
    total: float
    standalone: Dict[str, float]
    contributions: Dict[str, float]
    means: Dict[str, float]

    @property
    def total_mean(self) -> float:
        """The expected loss of the whole portfolio."""
        return math.fsum(self.means.values())

    @property
    def undiversified(self) -> float:
        """The sum of the standalone risk measures."""
        return math.fsum(self.standalone.values())

    @property
    def diversification_benefit(self) -> float:
        """``undiversified − total``. Never negative for TVaR; it can be for VaR, which is not subadditive."""
        return self.undiversified - self.total

    @property
    def diversification_ratio(self) -> float:
        """``1 − total/undiversified``."""
        return 1.0 - self.total / self.undiversified if self.undiversified else 0.0

    @property
    def shares(self) -> Dict[str, float]:
        """Each line's contribution as a fraction of the total (decimals that add up to 1)."""
        return {n: (c / self.total if self.total else 0.0) for n, c in self.contributions.items()}

    @property
    def excess(self) -> Dict[str, float]:
        """Each line's contribution above its own expected loss: capital in the "unexpected loss" sense.

        The values add up to ``total − total_mean``.
        """
        return {n: self.contributions[n] - self.means[n] for n in self.names}

    def to_dict(self) -> Dict[str, Any]:
        """Everything as plain JSON-able data."""
        return {
            "measure": self.measure, "method": self.method, "level": self.level, "trials": self.trials,
            "total": self.total, "total_mean": self.total_mean, "undiversified": self.undiversified,
            "diversification_benefit": self.diversification_benefit,
            "diversification_ratio": self.diversification_ratio,
            "standalone": dict(self.standalone), "contributions": dict(self.contributions),
            "shares": self.shares, "means": dict(self.means), "excess": self.excess,
        }

    def to_frame(self):
        """A pandas DataFrame indexed by line: mean, standalone, contribution, share, excess."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment dependent
            from .errors import MissingDependencyError

            raise MissingDependencyError("to_frame() needs pandas: pip install pandas", code="RP203") from exc
        shares, excess = self.shares, self.excess
        return pd.DataFrame(
            {"mean": [self.means[n] for n in self.names],
             "standalone": [self.standalone[n] for n in self.names],
             "contribution": [self.contributions[n] for n in self.names],
             "share": [shares[n] for n in self.names],
             "excess": [excess[n] for n in self.names]},
            index=list(self.names),
        )

    def summary(self) -> str:
        """A text table: mean, standalone, contribution and share per line, then the totals."""
        shares = self.shares
        label = "TVaR" if self.measure == "tvar" else "VaR"
        rows = [[n, _fmt_money(self.means[n]), _fmt_money(self.standalone[n]), _fmt_money(self.contributions[n]),
                 f"{shares[n]:.1%}"] for n in self.names]
        rows.append(["Total", _fmt_money(self.total_mean), _fmt_money(self.undiversified), _fmt_money(self.total),
                     "100.0%"])
        head = f"{label} {self.level:g} allocation ({self.method}, {self.trials:,} trials)"
        return (head + "\n" + _text_table(["Line", "Mean", f"Standalone {label}", "Contribution", "Share"], rows)
                + f"\nDiversification benefit {_fmt_money(self.diversification_benefit)} "
                  f"({self.diversification_ratio:.1%} of the standalone sum)")

    def plot(self, **kwargs):
        """The contributions as a bar chart (:func:`riskpy.viz.allocation`); needs Matplotlib.

        Keyword arguments (``title``, ``figsize``) go to the chart. Returns the Figure.

        Example::

            allocate(losses, level=0.995).plot(title="99.5% TVaR by line")
        """
        from . import viz

        label = "TVaR" if self.measure == "tvar" else "VaR"
        kwargs.setdefault("title", f"{label} {self.level:g} by line ({self.method})")
        return viz.allocation(self.contributions, total=self.total, **kwargs)

    def __repr__(self) -> str:
        return (f"<Allocation {self.measure} {self.level:g} total={self.total:.6g} lines={len(self.names)} "
                f"method={self.method}>")


def _tail_weights(values, level: float, np):
    """Weights that turn a mean over trials into the Acerbi–Tasche expected shortfall.

    Returns ``(q, weights)`` with ``q`` the lower empirical ``level``-quantile and
    ``Σ weights·values = ES``. Trials strictly beyond ``q`` weigh ``1/(n(1 − level))``; the
    trials sitting exactly on ``q`` share what is left of the tail probability equally, so
    an atom at the quantile is counted for the part of it that lies in the tail.
    """
    n = values.size
    ordered = np.sort(values)
    k = min(max(int(math.ceil(n * level - 1e-9)) - 1, 0), n - 1)
    q = float(ordered[k])
    tail = n * (1.0 - level)
    above = values > q
    at = values == q
    weights = np.zeros(n, dtype=float)
    weights[above] = 1.0
    weights[at] = max(tail - float(above.sum()), 0.0) / float(at.sum())
    return q, weights / weights.sum()


def _measure(values, level: float, measure: str, np) -> float:
    q, weights = _tail_weights(values, level, np)
    return q if measure == "var" else float(weights @ values)


def _loss_columns(losses: Any, names: Optional[Sequence[str]], np) -> Tuple[List[str], Any]:
    if hasattr(losses, "columns") and hasattr(losses, "to_numpy"):         # a pandas DataFrame
        labels = [str(c) for c in losses.columns]
        matrix = np.asarray(losses.to_numpy(), dtype=float)
    elif isinstance(losses, Mapping):
        if names is not None:
            raise InputError("pass names only when losses is an array", code="RP113", param="names")
        labels = [str(k) for k in losses]
        if not labels:
            raise InputError("losses is empty: give at least one line", code="RP111", param="losses")
        columns = [np.asarray(v, dtype=float) for v in losses.values()]
        for label, column in zip(labels, columns):
            if column.ndim != 1:
                raise InputError(f"losses[{label!r}] must be one-dimensional, got shape {column.shape}",
                                 code="RP100", param="losses")
            if column.size != columns[0].size:
                raise InputError(f"every line needs the same number of trials: {labels[0]!r} has "
                                 f"{columns[0].size}, {label!r} has {column.size}", code="RP109", param="losses",
                                 hint="The arrays must come from one joint simulation, trial for trial.")
        matrix = np.column_stack(columns)
    else:
        try:
            matrix = np.asarray(losses, dtype=float)
        except (TypeError, ValueError):
            raise ArgumentTypeError("losses must be {name: array} or a (trials, lines) array", code="RP404",
                                    param="losses") from None
        if matrix.ndim != 2:
            raise InputError(f"losses must be {{name: array}} or a 2-D (trials, lines) array, got shape "
                             f"{matrix.shape}", code="RP100", param="losses")
        if names is None:
            labels = [f"line_{i + 1}" for i in range(matrix.shape[1])]
        else:
            labels = [str(n) for n in names]
            if len(labels) != matrix.shape[1]:
                raise InputError(f"names has {len(labels)} entries but losses has {matrix.shape[1]} columns",
                                 code="RP109", param="names")
    if matrix.shape[1] < 1:
        raise InputError("losses is empty: give at least one line", code="RP111", param="losses")
    if len(set(labels)) != len(labels):
        raise InputError("line names must be distinct", code="RP113", param="names")
    if matrix.shape[0] < 2:
        raise InputError(f"at least two trials are needed, got {matrix.shape[0]}", code="RP111", param="losses")
    if not np.all(np.isfinite(matrix)):
        raise InputError("losses must be finite (no NaN or infinity)", code="RP101", param="losses")
    return labels, matrix


def allocate(losses: Any, level: float = 0.99, measure: str = "tvar", method: str = "euler",
             names: Optional[Sequence[str]] = None, bandwidth: Optional[float] = None) -> Allocation:
    """Measure the capital of a portfolio of simulated losses and split it between its lines.

    losses: ``{name: array}`` — one array of simulated losses per line, all from the same
        joint simulation so that entry ``k`` of every array is the same trial (what
        ``Model.sample(...)`` returns) — or a ``(trials, lines)`` array or DataFrame. Losses
        are positive numbers: bigger is worse.
    level: confidence level as a decimal, e.g. 0.99
    measure: "tvar" (expected shortfall, the default) or "var"
    method: how the total is split —

        - "euler" (default): each line's marginal contribution. For TVaR this is the
          co-TVaR ``E[Xᵢ | S in the worst (1 − level)]``, an exact identity on the sample.
          For VaR it is ``E[Xᵢ | S = VaR]``, estimated from the trials whose total is
          nearest the VaR and scaled to add up; it is much noisier than co-TVaR.
        - "covariance": ``E[Xᵢ] + Cov(Xᵢ, S)/Var(S) · (total − E[S])``. Equal to Euler when
          the losses are jointly normal; stable, but blind to which line fills the tail.
        - "standalone": pro rata to each line's own risk measure.
    names: labels for the columns when ``losses`` is a plain array
    bandwidth: for Euler VaR only, the share of trials around the VaR to average over
        (default ``trials^(−1/2)``, at least 5 trials each side)

    Returns an :class:`Allocation`. ``total`` is the risk measure of the summed losses and
    equals ``Result.tvar(level)`` / the lower empirical quantile for the same sample;
    ``contributions`` add up to it; ``standalone``, ``diversification_benefit``, ``shares``
    and ``excess`` (contribution above the line's mean) follow.

    The Euler principle is the allocation under which a line's return on allocated capital
    moves the same way as the portfolio's when the line grows — the only one that does not
    send the wrong signal to whoever prices the business.

    Example::

        from riskpy.mc import Model, LogNormal
        from riskpy.capital import GumbelCopula, allocate

        book = Model(motor=LogNormal.from_moments(100, 20), property=LogNormal.from_moments(60, 30))
        losses = book.sample(200_000, seed=1, copula=GumbelCopula(1.5))
        alloc = allocate(losses, level=0.99)
        alloc.total, alloc.contributions, alloc.diversification_benefit
        print(alloc.summary())
    """
    np = _numpy()
    level = _check_level(level)
    if measure not in ("tvar", "var"):
        raise InputError(f"measure must be 'tvar' or 'var', got {measure!r}", code="RP108", param="measure")
    if method not in ("euler", "covariance", "standalone"):
        raise InputError(f"method must be 'euler', 'covariance' or 'standalone', got {method!r}", code="RP108",
                         param="method")
    labels, matrix = _loss_columns(losses, names, np)
    n, k = matrix.shape
    total_losses = matrix.sum(axis=1)
    means = matrix.mean(axis=0)
    quantile, weights = _tail_weights(total_losses, level, np)
    total = quantile if measure == "var" else float(weights @ total_losses)
    standalone = [_measure(matrix[:, j], level, measure, np) for j in range(k)]

    if method == "euler" and measure == "tvar":
        contributions = weights @ matrix
    elif method == "euler":
        if bandwidth is None:
            half = max(int(round(0.5 * math.sqrt(n))), 5)
        else:
            if not _is_number(bandwidth) or not 0.0 < bandwidth <= 1.0:
                raise InputError(f"bandwidth must be in (0, 1], got {bandwidth!r}", code="RP100", param="bandwidth")
            half = max(int(round(0.5 * bandwidth * n)), 1)
        order = np.argsort(total_losses, kind="mergesort")
        centre = min(max(int(math.ceil(n * level - 1e-9)) - 1, 0), n - 1)
        half = min(half, centre, n - 1 - centre)          # a symmetric window, so its mean total stays at the VaR
        window = order[centre - half:centre + half + 1]
        local = matrix[window].mean(axis=0)
        local_total = float(local.sum())
        # Scale the window mean so the pieces add up to the VaR itself.
        contributions = local * (total / local_total) if local_total != 0.0 else np.full(k, total / k)
    elif method == "covariance":
        centred = matrix - means
        s_centred = total_losses - total_losses.mean()
        variance = float(s_centred @ s_centred)
        if variance > 0.0:
            beta = (centred.T @ s_centred) / variance
        else:
            beta = np.full(k, 1.0 / k)
        contributions = means + beta * (total - float(total_losses.mean()))
    else:
        base = float(sum(standalone))
        if base != 0.0:
            contributions = np.array(standalone) * (total / base)
        else:
            contributions = np.full(k, total / k)

    return Allocation(
        names=tuple(labels), level=level, measure=measure, method=method, trials=int(n), total=float(total),
        standalone={label: float(standalone[j]) for j, label in enumerate(labels)},
        contributions={label: float(contributions[j]) for j, label in enumerate(labels)},
        means={label: float(means[j]) for j, label in enumerate(labels)},
    )


# ---------------------------------------------------------------------------
# Verification


def _verification_checks():
    """Return a list of (name, value, reference, tolerance) tuples."""
    checks = []
    # Kendall's tau and tail dependence: closed forms worked by hand.
    checks.append(("Gaussian copula: tau = 2/π·asin(ρ) = 1/3 at ρ = 0.5",
                   GaussianCopula(0.5).kendall_tau(), 1.0 / 3.0, 1e-15))
    checks.append(("Gaussian copula: C(½, ½) = 1/4 + asin(ρ)/(2π) (Sheppard)",
                   GaussianCopula(0.3).cdf(0.5, 0.5), 0.25 + math.asin(0.3) / (2.0 * math.pi), 1e-12))
    checks.append(("t copula: C(½, ½) is Sheppard's value too",
                   StudentTCopula(-0.6, df=3).cdf(0.5, 0.5), 0.25 + math.asin(-0.6) / (2.0 * math.pi), 1e-12))
    # ν = 1, ρ = 0: λ = 2·t₂(−√2), and t₂(x) = ½ + x/(2√(2 + x²)), so λ = 1 − 1/√2.
    checks.append(("t copula: tail dependence 1 − 1/√2 at df = 1, ρ = 0",
                   StudentTCopula(0.0, df=1).tail_dependence().upper, 1.0 - 1.0 / math.sqrt(2.0), 1e-12))
    checks.append(("Clayton copula: lower tail dependence 2^(−1/θ) = 1/√2 at θ = 2",
                   ClaytonCopula(2.0).tail_dependence().lower, 1.0 / math.sqrt(2.0), 1e-15))
    checks.append(("Gumbel copula: C(u, u) = u^(2^(1/θ))",
                   GumbelCopula(2.0).cdf(0.9, 0.9), 0.9 ** math.sqrt(2.0), 1e-14))
    checks.append(("Frank copula: tau → θ/9 for small θ", FrankCopula(1e-3).kendall_tau(), 1e-3 / 9.0, 1e-9))
    # Variance–covariance aggregation: 3-4-5 triangle for independent risks.
    agg = aggregate_capital([3.0, 4.0], [[1.0, 0.0], [0.0, 1.0]])
    checks.append(("aggregate_capital: independent 3 and 4 give 5", agg.total, 5.0, 1e-14))
    checks.append(("aggregate_capital: Euler contributions 9/5 + 16/5",
                   agg.contributions["risk_1"] - 1.8 + agg.contributions["risk_2"] - 3.2, 0.0, 1e-14))
    try:
        np = _numpy()
    except ImportError:  # pragma: no cover - NumPy absent
        return checks
    fixed = nearest_correlation([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    checks.append(("nearest_correlation: Higham's (2002) example, off-diagonal 0.7607", fixed[0, 1], 0.7607, 5e-5))
    checks.append(("nearest_correlation: Higham's (2002) example, corner 0.1573", fixed[0, 2], 0.1573, 5e-5))
    # Euler TVaR for jointly normal losses: μᵢ + Cov(Xᵢ, S)/σ_S · φ(z)/(1 − α).
    rho, level, trials = 0.5, 0.99, 400_000
    u = GaussianCopula(rho).sample(trials, seed=11)
    x = 10.0 + 2.0 * _special.norm_ppf_vec(u[:, 0])
    y = 5.0 + 1.0 * _special.norm_ppf_vec(u[:, 1])
    alloc = allocate({"x": x, "y": y}, level=level)
    sd_total = math.sqrt(4.0 + 1.0 + 2.0 * rho * 2.0 * 1.0)
    ratio = _special.norm_pdf(_special.norm_ppf(level)) / (1.0 - level)
    reference = 10.0 + (4.0 + rho * 2.0) / sd_total * ratio
    checks.append(("allocate: Euler TVaR contribution of a normal line (400k trials)",
                   alloc.contributions["x"], reference, 0.1))
    checks.append(("allocate: contributions add up to the total TVaR",
                   alloc.contributions["x"] + alloc.contributions["y"], alloc.total, 1e-9))
    tau_hat = _sample_kendall(GumbelCopula(2.0).sample(4_000, seed=5), np)
    checks.append(("Gumbel sampler: Kendall's tau 1 − 1/θ (4,000 draws)", tau_hat, 0.5, 0.05))
    return checks


def _sample_kendall(u, np) -> float:
    """Kendall's tau of a two-column sample by the O(n²) definition (verification only)."""
    a, b = u[:, 0], u[:, 1]
    s = 0.0
    for i in range(len(a) - 1):
        s += float(np.sum(np.sign(a[i + 1:] - a[i]) * np.sign(b[i + 1:] - b[i])))
    n = len(a)
    return s / (0.5 * n * (n - 1))
