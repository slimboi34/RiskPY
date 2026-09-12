"""Special functions and small numerical routines, dependency-free.

Everything the modelling layers need from SciPy, written out so the package
does not have to depend on it: incomplete gamma and beta functions, the normal,
Student-t and chi-square distributions, root finding, a bounded minimiser, a
Nelder–Mead simplex, and Simpson integration.

Each routine notes its accuracy. They are checked against SciPy in the test
suite (SciPy is a test-time oracle only), and again in ``riskpy.verify``.

Scalar functions work on floats. The ``*_vec`` variants take NumPy arrays and
are what the distribution classes in :mod:`riskpy.mc` use, so a CDF on 200,000
points is one vectorised pass rather than a Python loop.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "norm_cdf",
    "norm_pdf",
    "norm_ppf",
    "gammaln",
    "digamma",
    "trigamma",
    "gammainc",
    "gammaincc",
    "gammainc_vec",
    "gammaincc_vec",
    "betainc",
    "beta_function",
    "t_cdf",
    "t_ppf",
    "chi2_cdf",
    "chi2_ppf",
    "kolmogorov_sf",
    "ks_pvalue",
    "brent",
    "bisect",
    "minimise_scalar",
    "nelder_mead",
    "simpson",
    "invert_monotone",
]

_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_EPS = 2.220446049250313e-16
_FPMIN = 1e-300


# ---------------------------------------------------------------------------
# Normal distribution
# ---------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    """Standard normal CDF via ``erfc``, so the far tails do not underflow.

    ``0.5 * (1 + erf(x/√2))`` loses everything below about ``x = -8`` because
    ``erf`` rounds to -1; ``erfc`` keeps its relative precision down to
    ``x ≈ -37``, where it finally underflows. That matters for VaR at 99.99% and for likelihoods.
    """
    return 0.5 * math.erfc(-x / _SQRT2)


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF, full double precision.

    Acklam's rational approximation (relative error 1.15e-9) followed by one
    Halley refinement step against ``erfc``, which takes it to machine
    precision everywhere in (0, 1).
    """
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if p > 0.5:
        # Work in the lower tail, where ``erfc`` keeps its relative precision;
        # refining directly against ``1 - tiny`` loses the tail digits.
        return -norm_ppf(1.0 - p)

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)

    low, high = 0.02425, 1.0 - 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p > high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

    # One Halley step: e is the CDF error, u = e / pdf(x).
    e = 0.5 * math.erfc(-x / _SQRT2) - p
    u = e * _SQRT_2PI * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


# ---------------------------------------------------------------------------
# Gamma family
# ---------------------------------------------------------------------------


def gammaln(x: float) -> float:
    """log Γ(x) for ``x > 0`` — ``math.lgamma`` with a clearer name."""
    if x <= 0.0:
        raise ValueError(f"gammaln needs x > 0, got {x!r}")
    return math.lgamma(x)


def digamma(x: float) -> float:
    """ψ(x) = d/dx log Γ(x). Recurrence up to x ≥ 10, then the asymptotic series.

    Absolute error below 1e-13 for ``x > 0``.
    """
    if x <= 0.0:
        raise ValueError(f"digamma needs x > 0, got {x!r}")
    result = 0.0
    while x < 10.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += math.log(x) - 0.5 * inv - inv2 * (
        1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 / 132.0)))
    )
    return result


def trigamma(x: float) -> float:
    """ψ₁(x) = d²/dx² log Γ(x). Same construction as :func:`digamma`."""
    if x <= 0.0:
        raise ValueError(f"trigamma needs x > 0, got {x!r}")
    result = 0.0
    while x < 10.0:
        result += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += inv + 0.5 * inv2 + inv * inv2 * (
        1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 * (1.0 / 30.0 - inv2 * 5.0 / 66.0)))
    )
    return result


def _gser(a: float, x: float) -> float:
    """Series for the regularised lower incomplete gamma, x < a + 1."""
    ap = a
    total = term = 1.0 / a
    for _ in range(1000):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    """Continued fraction (modified Lentz) for the regularised upper incomplete gamma, x ≥ a + 1."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) = γ(a, x) / Γ(a).

    Series for ``x < a + 1``, continued fraction otherwise (Numerical Recipes
    6.2). Relative error around 1e-14. This is the Gamma CDF, the Poisson
    survival function and the chi-square CDF in one routine.
    """
    if a <= 0.0:
        raise ValueError(f"gammainc needs a > 0, got a={a!r}")
    if x < 0.0:
        raise ValueError(f"gammainc needs x >= 0, got x={x!r}")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


def gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) = 1 - P(a, x), computed
    directly in the tail so it does not lose precision to cancellation."""
    if a <= 0.0:
        raise ValueError(f"gammaincc needs a > 0, got a={a!r}")
    if x < 0.0:
        raise ValueError(f"gammaincc needs x >= 0, got x={x!r}")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


# ---------------------------------------------------------------------------
# Beta family
# ---------------------------------------------------------------------------


def beta_function(a: float, b: float) -> float:
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


def _betacf(a: float, b: float, x: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) — the Beta CDF, and through it the
    Student-t, F, binomial and negative binomial CDFs.

    Continued fraction (Numerical Recipes 6.4) with the symmetry relation to
    keep it convergent. Relative error around 1e-14.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"betainc needs a, b > 0, got a={a!r} b={b!r}")
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"betainc needs 0 <= x <= 1, got x={x!r}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


# ---------------------------------------------------------------------------
# Derived distributions
# ---------------------------------------------------------------------------


def t_cdf(x: float, df: float) -> float:
    """Student-t CDF with ``df`` degrees of freedom (need not be an integer)."""
    if df <= 0.0:
        raise ValueError(f"df must be > 0, got {df!r}")
    if x == 0.0:
        return 0.5
    t2 = x * x
    ib = betainc(0.5 * df, 0.5, df / (df + t2))
    return 1.0 - 0.5 * ib if x > 0.0 else 0.5 * ib


def t_ppf(p: float, df: float) -> float:
    """Student-t quantile, by bracketed root finding on :func:`t_cdf`."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    if df <= 0.0:
        raise ValueError(f"df must be > 0, got {df!r}")
    if p == 0.5:
        return 0.0
    if df > 1e6:
        return norm_ppf(p)
    # Bracket: t quantiles are wider than normal ones, and grow like p^(-1/df).
    guess = abs(norm_ppf(p)) if 0.02 < p < 0.98 else 1.0
    hi = max(4.0 * guess, 2.0)
    while t_cdf(hi, df) < max(p, 1.0 - p):
        hi *= 2.0
        if hi > 1e300:
            raise ValueError("t_ppf failed to bracket the quantile")
    target = max(p, 1.0 - p)
    value = brent(lambda t: t_cdf(t, df) - target, 0.0, hi)
    return value if p > 0.5 else -value


def chi2_cdf(x: float, k: float) -> float:
    """Chi-square CDF with ``k`` degrees of freedom."""
    if k <= 0.0:
        raise ValueError(f"k must be > 0, got {k!r}")
    if x <= 0.0:
        return 0.0
    return gammainc(0.5 * k, 0.5 * x)


def chi2_ppf(p: float, k: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    hi = max(10.0, 4.0 * k)
    while chi2_cdf(hi, k) < p:
        hi *= 2.0
    return brent(lambda x: chi2_cdf(x, k) - p, 0.0, hi)


def kolmogorov_sf(x: float) -> float:
    """P(K > x) for the Kolmogorov distribution, 2 Σ (-1)^{k-1} exp(-2 k² x²)."""
    if x <= 0.0:
        return 1.0
    if x > 6.0:
        return 0.0
    total = 0.0
    for k in range(1, 200):
        term = 2.0 * (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * x * x)
        total += term
        if abs(term) < 1e-16:
            break
    return min(1.0, max(0.0, total))


def ks_pvalue(statistic: float, n: int) -> float:
    """Asymptotic Kolmogorov–Smirnov p-value with Stephens' small-sample
    correction, ``(√n + 0.12 + 0.11/√n) · D``. Good to a couple of decimals for
    ``n ≥ 20``, which is all a goodness-of-fit ranking needs."""
    if n < 1:
        raise ValueError("n must be >= 1")
    root = math.sqrt(n)
    return kolmogorov_sf((root + 0.12 + 0.11 / root) * statistic)


# ---------------------------------------------------------------------------
# Root finding, minimisation, integration
# ---------------------------------------------------------------------------


def bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-12, max_iter: int = 500) -> float:
    """Plain bisection — slow and unkillable. Requires a sign change."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError(f"no sign change on [{lo}, {hi}]: f = {flo:.6g}, {fhi:.6g}")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0.0 or 0.5 * (hi - lo) < tol:
            return mid
        if flo * fmid < 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def brent(f: Callable[[float], float], a: float, b: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Brent's root finder on a bracket ``[a, b]`` with a sign change.

    Superlinear where the function is smooth, and it never leaves the bracket,
    so it is the right default for implied volatility, yields and quantiles.
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(f"no sign change on [{a}, {b}]: f = {fa:.6g}, {fb:.6g}")
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa
    c, fc = a, fa
    mflag = True
    d = 0.0
    for _ in range(max_iter):
        if fb == 0.0 or abs(b - a) < tol:
            return b
        if fa != fc and fb != fc:
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa)
        cond1 = not ((3.0 * a + b) / 4.0 < s < b or b < s < (3.0 * a + b) / 4.0)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2.0
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2.0
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol
        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False
        fs = f(s)
        d, c, fc = c, b, fb
        if fa * fs < 0.0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa
    return b


def invert_monotone(f: Callable[[float], float], target: float, lo: float, hi: float,
                    tol: float = 1e-12, expand: bool = True) -> float:
    """Solve ``f(x) = target`` for an increasing ``f`` on ``[lo, hi]``,
    widening the bracket upward if ``expand`` and the target is not yet covered."""
    if expand:
        for _ in range(200):
            if f(hi) >= target:
                break
            lo, hi = hi, hi * 2.0 if hi > 0 else hi + 1.0
        else:
            raise ValueError("invert_monotone could not bracket the target")
    return brent(lambda x: f(x) - target, lo, hi, tol=tol)


def minimise_scalar(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-10, max_iter: int = 500) -> float:
    """Bounded scalar minimiser (Brent's golden-section / parabolic method).
    Returns the location of the minimum of a unimodal ``f`` on ``[lo, hi]``."""
    golden = 0.3819660112501051
    a, b = lo, hi
    x = w = v = a + golden * (b - a)
    fx = fw = fv = f(x)
    d = e = 0.0
    for _ in range(max_iter):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-12
        tol2 = 2.0 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            return x
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp = e
            e = d
            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x):
                e = (a - x) if x >= xm else (b - x)
                d = golden * e
            else:
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if xm - x >= 0 else -tol1
        else:
            e = (a - x) if x >= xm else (b - x)
            d = golden * e
        u = x + (d if abs(d) >= tol1 else (tol1 if d >= 0 else -tol1))
        fu = f(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def nelder_mead(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    step: float | Sequence[float] = 0.1,
    tol: float = 1e-10,
    max_iter: int = 5000,
) -> Tuple[List[float], float]:
    """Nelder–Mead simplex minimiser. Returns ``(x_min, f_min)``.

    Derivative-free and robust, which is what maximum-likelihood fits of
    two- and three-parameter distributions need. Restart it from the answer if
    you suspect a premature stop — it is cheap.
    """
    n = len(x0)
    if n == 0:
        raise ValueError("x0 must have at least one element")
    steps = [step] * n if isinstance(step, (int, float)) else list(step)
    if len(steps) != n:
        raise ValueError("step must be a scalar or match the length of x0")

    simplex = [list(map(float, x0))]
    for i in range(n):
        point = list(map(float, x0))
        point[i] += steps[i] if steps[i] != 0 else 0.05
        simplex.append(point)
    values = [f(p) for p in simplex]

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda k: values[k])
        simplex = [simplex[k] for k in order]
        values = [values[k] for k in order]

        spread = max(abs(values[i] - values[0]) for i in range(1, n + 1))
        size = max(max(abs(simplex[i][j] - simplex[0][j]) for j in range(n)) for i in range(1, n + 1))
        if spread <= tol * (1.0 + abs(values[0])) and size <= tol * (1.0 + max(abs(v) for v in simplex[0])):
            break

        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        worst = simplex[-1]
        reflected = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(n)]
        f_reflected = f(reflected)

        if values[0] <= f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
            continue
        if f_reflected < values[0]:
            expanded = [centroid[j] + gamma * (reflected[j] - centroid[j]) for j in range(n)]
            f_expanded = f(expanded)
            if f_expanded < f_reflected:
                simplex[-1], values[-1] = expanded, f_expanded
            else:
                simplex[-1], values[-1] = reflected, f_reflected
            continue
        contracted = [centroid[j] + rho * (worst[j] - centroid[j]) for j in range(n)]
        f_contracted = f(contracted)
        if f_contracted < values[-1]:
            simplex[-1], values[-1] = contracted, f_contracted
            continue
        best = simplex[0]
        for i in range(1, n + 1):
            simplex[i] = [best[j] + sigma * (simplex[i][j] - best[j]) for j in range(n)]
            values[i] = f(simplex[i])

    best_index = min(range(n + 1), key=lambda k: values[k])
    return simplex[best_index], values[best_index]


def simpson(f: Callable[[float], float], a: float, b: float, n: int = 200) -> float:
    """Composite Simpson's rule with ``n`` (even) intervals."""
    if n < 2:
        raise ValueError("n must be >= 2")
    if n % 2:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * f(a + i * h)
    return total * h / 3.0


# ---------------------------------------------------------------------------
# Vectorised variants (NumPy). Same algorithms with fixed iteration counts and
# masks, so a CDF over a large array is one pass.
# ---------------------------------------------------------------------------


def gammainc_vec(a: float, x):
    """Regularised lower incomplete gamma over a NumPy array ``x``, scalar ``a``.

    Same series and continued fraction as :func:`gammainc`, but each iteration
    carries only the points that have not converged yet. Without that, a batch
    spanning the whole range runs every iteration the slowest point needs — for
    a non-integer shape that is the 500-iteration cap, and the array CDF ends
    up two orders of magnitude slower than it has any reason to be.
    """
    return _gamma_pq_vec(a, x, upper=False)


def gammaincc_vec(a: float, x):
    """Regularised upper incomplete gamma ``Q(a, x)`` over an array, computed
    directly in the tail so that ``Q = 1e-12`` keeps its twelve digits instead
    of the three that ``1 - P`` would leave it."""
    return _gamma_pq_vec(a, x, upper=True)


def _gamma_pq_vec(a: float, x, upper: bool):
    import numpy as np

    if a <= 0.0:
        raise ValueError(f"gammainc needs a > 0, got a={a!r}")
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("gammainc needs x >= 0")
    shape = x.shape
    flat = x.ravel()
    out = np.zeros(flat.size, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        prefix = np.exp(-flat + a * np.log(np.where(flat > 0, flat, 1.0)) - math.lgamma(a))

    series_idx = np.flatnonzero((flat > 0) & (flat < a + 1.0))
    if series_idx.size:
        xs = flat[series_idx]
        term = np.full(xs.shape, 1.0 / a)
        total = term.copy()
        idx = np.arange(xs.size)
        ap = a
        for _ in range(1000):
            if idx.size == 0:
                break
            ap += 1.0
            term_i = term[idx] * xs[idx] / ap
            total_i = total[idx] + term_i
            term[idx], total[idx] = term_i, total_i
            idx = idx[np.abs(term_i) >= np.abs(total_i) * _EPS]
        lower = total * prefix[series_idx]
        out[series_idx] = 1.0 - lower if upper else lower

    cf_idx = np.flatnonzero(flat >= a + 1.0)
    if cf_idx.size:
        xc = flat[cf_idx]
        b = xc + 1.0 - a
        c = np.full(xc.shape, 1.0 / _FPMIN)
        d = 1.0 / b
        h = d.copy()
        idx = np.arange(xc.size)
        for i in range(1, 1000):
            if idx.size == 0:
                break
            an = -i * (i - a)
            b_i = b[idx] + 2.0
            d_i = an * d[idx] + b_i
            d_i = np.where(np.abs(d_i) < _FPMIN, _FPMIN, d_i)
            c_i = b_i + an / c[idx]
            c_i = np.where(np.abs(c_i) < _FPMIN, _FPMIN, c_i)
            d_i = 1.0 / d_i
            delta = d_i * c_i
            b[idx], c[idx], d[idx] = b_i, c_i, d_i
            h[idx] = h[idx] * delta
            idx = idx[np.abs(delta - 1.0) >= _EPS]
        tail = prefix[cf_idx] * h
        out[cf_idx] = tail if upper else 1.0 - tail

    if upper:
        out[flat == 0.0] = 1.0
    return np.clip(out, 0.0, 1.0).reshape(shape)


def _betacf_vec(a: float, b: float, x):
    """Modified-Lentz continued fraction for the incomplete beta, over an array,
    iterating only the points that have not converged."""
    import numpy as np

    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones(x.size, dtype=float)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < _FPMIN, _FPMIN, d)
    d = 1.0 / d
    h = d.copy()
    idx = np.arange(x.size)
    for m in range(1, 1000):
        if idx.size == 0:
            break
        m2 = 2 * m
        xi = x[idx]
        ci, di = c[idx], d[idx]

        num = m * (b - m) * xi / ((qam + m2) * (a + m2))
        di = 1.0 + num * di
        di = np.where(np.abs(di) < _FPMIN, _FPMIN, di)
        ci = 1.0 + num / ci
        ci = np.where(np.abs(ci) < _FPMIN, _FPMIN, ci)
        di = 1.0 / di
        hi = h[idx] * di * ci

        num = -(a + m) * (qab + m) * xi / ((a + m2) * (qap + m2))
        di = 1.0 + num * di
        di = np.where(np.abs(di) < _FPMIN, _FPMIN, di)
        ci = 1.0 + num / ci
        ci = np.where(np.abs(ci) < _FPMIN, _FPMIN, ci)
        di = 1.0 / di
        delta = di * ci
        hi = hi * delta

        c[idx], d[idx], h[idx] = ci, di, hi
        idx = idx[np.abs(delta - 1.0) >= _EPS]
    return h


def betainc_vec(a: float, b: float, x):
    """Regularised incomplete beta over a NumPy array ``x``, scalar ``a, b``."""
    import numpy as np

    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"betainc needs a, b > 0, got a={a!r} b={b!r}")
    x = np.asarray(x, dtype=float)
    if np.any((x < 0) | (x > 1)):
        raise ValueError("betainc needs 0 <= x <= 1")
    shape = x.shape
    flat = x.ravel()
    out = np.empty(flat.size, dtype=float)
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)

    out[flat <= 0.0] = 0.0
    out[flat >= 1.0] = 1.0
    inner = np.flatnonzero((flat > 0.0) & (flat < 1.0))
    if inner.size:
        xi = flat[inner]
        front = np.exp(lbeta + a * np.log(xi) + b * np.log1p(-xi))
        direct = xi < (a + 1.0) / (a + b + 2.0)
        res = np.empty(xi.size, dtype=float)
        fwd = np.flatnonzero(direct)
        if fwd.size:
            res[fwd] = front[fwd] * _betacf_vec(a, b, xi[fwd]) / a
        rev = np.flatnonzero(~direct)
        if rev.size:
            res[rev] = 1.0 - front[rev] * _betacf_vec(b, a, 1.0 - xi[rev]) / b
        out[inner] = res
    return np.clip(out, 0.0, 1.0).reshape(shape)


_VEC_ERFC = None
_VEC_PPF = None


def norm_cdf_vec(x):
    """Standard normal CDF over a NumPy array, keeping the ``erfc`` tail.

    NumPy has no ``erfc``, and a rational approximation would give up the far
    tail that :func:`norm_cdf` exists to protect, so this maps ``math.erfc``
    over the array with ``frompyfunc`` — a C-level loop, about 50 ns a point.
    """
    import numpy as np

    global _VEC_ERFC
    if _VEC_ERFC is None:
        _VEC_ERFC = np.frompyfunc(math.erfc, 1, 1)
    x = np.asarray(x, dtype=float)
    return np.asarray(_VEC_ERFC(-x / _SQRT2), dtype=float) * 0.5


def norm_ppf_vec(p):
    """Inverse standard normal CDF over a NumPy array."""
    import numpy as np

    global _VEC_PPF
    if _VEC_PPF is None:
        _VEC_PPF = np.frompyfunc(norm_ppf, 1, 1)
    p = np.asarray(p, dtype=float)
    return np.asarray(_VEC_PPF(p), dtype=float)
