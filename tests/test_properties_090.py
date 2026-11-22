"""Hypothesis property tests from the machine-driven testing of RiskPY 0.8.0 (added in 0.9.0).

Three sections, each with its own fixed, derandomized Hypothesis profile (no example database, no
deadline), so the file is deterministic and runs in well under a minute:

1. numerical properties against independent oracles (mpmath, closed forms written from scratch);
2. the audit ledger: round trips, tamper detection, continuation, threads and processes;
3. the agent tool surface: schema-generated arguments, LIMITS, and metamorphic checks.

The campaign confirmed 21 defects in 0.8.0. The tests that pinned them as strict xfails now pass and carry
a "finding N" comment instead; finding 20 (a ledger re-serialised with numbers re-spelled) is a documented
limit and is pinned as such. RISKPY_PROPS_CAMPAIGN=<factor> turns the file into a randomized campaign with
<factor> times as many examples.
"""
from __future__ import annotations


# ==============================================================================================================
# Numerical properties against independent oracles (workstream 2)
# (from test_num_props.py)
# ==============================================================================================================

"""Property tests for RiskPY's numerics against independent oracles.

Oracles are mpmath at 30-40 digits (special functions, copula closed forms, Black-Scholes),
hand calculations written here (chain ladder, annuities, bonds, Basel IRB, expected
shortfall), and identities that must hold exactly (put-call parity, Wald, Euler, Frechet).

Deterministic by default: every test runs under a derandomized Hypothesis profile with a
modest number of examples (about 20 s for the whole file). Set
RISKPY_PROPS_CAMPAIGN=<factor> to run the same properties as a randomized campaign with
<factor> times as many examples.

Tests that pinned confirmed 0.8.0 defects as strict xfails now pass; each names its finding.
"""

import math
import os

import pytest

np = pytest.importorskip("numpy")
mp = pytest.importorskip("mpmath")
stats = pytest.importorskip("scipy.stats")
from hypothesis import HealthCheck, assume, given, note, settings, strategies as st  # noqa: E402

from riskpy import _special as S  # noqa: E402
from riskpy import credit, life, mc, quant, rates, reserving  # noqa: E402
from riskpy.capital import (  # noqa: E402
    ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula, StudentTCopula, allocate, nearest_correlation,
)

_CAMPAIGN = int(os.environ.get("RISKPY_PROPS_CAMPAIGN", "0"))     # a multiplier on each test's examples


def profile(examples: int = 40):
    if _CAMPAIGN:
        return settings(max_examples=examples * _CAMPAIGN, deadline=None, derandomize=False,
                        suppress_health_check=list(HealthCheck), database=None)
    return settings(max_examples=examples, deadline=None, derandomize=True,
                    suppress_health_check=list(HealthCheck), database=None)


mp.mp.dps = 40


def log_uniform(lo: float, hi: float):
    """Floats spread evenly in log10 between 10**lo and 10**hi."""
    return st.floats(min_value=lo, max_value=hi, allow_nan=False).map(lambda e: 10.0 ** e)


def rel(got: float, exact) -> float:
    exact = float(exact)
    return abs(got - exact) / abs(exact) if exact != 0.0 else abs(got)


# ---------------------------------------------------------------------------
# riskpy._special against mpmath


@profile(200)
@given(st.floats(min_value=-37.5, max_value=37.5))
def test_norm_cdf_matches_mpmath(x):
    # The relative condition number of Phi at x is about x^2 (rounding x itself costs that
    # much), so the bound grows with x^2; below x = -37.5 the result is subnormal.
    assert rel(S.norm_cdf(x), mp.ncdf(x)) < 4e-16 * (1.0 + x * x)


def _norm_ppf_mp(p: float):
    if p > 1e-10:
        return mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1)
    return mp.findroot(lambda z: mp.log(mp.ncdf(z)) - mp.log(mp.mpf(p)), -math.sqrt(-2.0 * math.log(p)))


@profile(150)
@given(st.one_of(log_uniform(-307.0, math.log10(0.5)), st.floats(min_value=1e-6, max_value=1 - 1e-12)))
def test_norm_ppf_matches_mpmath(p):
    assume(0.0 < p < 1.0)
    exact = _norm_ppf_mp(p)
    # relative 1e-13 everywhere: since 0.9.0 (finding 13) that includes the small quantiles near p = 0.5, which
    # needed 3e-16 of absolute slack in 0.8.0 (the campaign found 4e-13 relative at p = 0.50012)
    assert abs(S.norm_ppf(p) - float(exact)) <= 1e-13 * abs(float(exact))


# 0.8.0 machine testing, finding 13 (fixed in 0.9.0): norm_ppf was 1.1e-9 relative near p = 0.5.
def test_norm_ppf_keeps_relative_precision_near_one_half():
    for p in (0.5 - 1e-12, 0.5 + 1e-10):
        assert rel(S.norm_ppf(p), _norm_ppf_mp(p)) < 1e-14


def _t_log_density(t, df):
    t, df = mp.mpf(t), mp.mpf(df)
    return mp.loggamma((df + 1) / 2) - mp.loggamma(df / 2) - mp.log(df * mp.pi) / 2 - (df + 1) / 2 * mp.log1p(t * t / df)


def _t_tail_mp(x: float, df: float):
    """P(T < -|x|) in mpmath: the incomplete beta, or the density integrated if that fails to converge."""
    x, df = mp.mpf(abs(x)), mp.mpf(df)
    try:
        return mp.betainc(df / 2, mp.mpf(1) / 2, 0, df / (df + x * x), regularized=True) / 2
    except (ValueError, ZeroDivisionError, mp.libmp.NoConvergence):
        pts = [-mp.inf] + [-x - w for w in (100, 30, 10, 3, 1, 0.3, 0.1)] + [-x]
        return mp.quad(lambda t: mp.exp(_t_log_density(t, df)), pts)


@profile(120)
@given(df=log_uniform(-1.0, 9.0), x=log_uniform(-8.0, 3.0))
def test_t_cdf_lower_tail_matches_mpmath(df, x):
    tail = _t_tail_mp(x, df)
    assume(tail > 1e-290)
    assert rel(S.t_cdf(-x, df), tail) < 1e-11
    assert abs(S.t_cdf(x, df) - float(1 - tail)) < 1e-14


@profile(80)
@given(df=log_uniform(-0.5, 9.0), delta=log_uniform(-15.0, -0.7), upper=st.booleans())
def test_t_ppf_near_one_half_keeps_relative_accuracy(df, delta, upper):
    """Near p = 0.5 the quantile is tiny; SciPy loses it, so the check is P(0 < T < t) in mpmath."""
    p = 0.5 + delta if upper else 0.5 - delta
    t = S.t_ppf(p, df)
    assert (t > 0) == upper
    tt, nu = mp.mpf(t), mp.mpf(df)
    mass = mp.betainc(mp.mpf(1) / 2, nu / 2, 0, tt * tt / (nu + tt * tt), regularized=True) / 2   # P(0 < T < |t|)
    assert rel(float(mass), abs(mp.mpf(p) - mp.mpf("0.5"))) < 1e-12


@profile(80)
@given(df=log_uniform(-0.3, 9.0), p=log_uniform(-250.0, -2.0))
def test_t_ppf_far_tail_matches_mpmath(df, p):
    t = S.t_ppf(p, df)
    assume(math.isfinite(t))
    tail = _t_tail_mp(t, df)
    # Turn the error in the tail mass into a relative error in t: d ln(tail) / d ln|t| = |t| f(t) / tail.
    elasticity = abs(t) * mp.exp(_t_log_density(t, df)) / tail
    assert rel(float(tail), p) / float(elasticity) < 1e-10


@profile(100)
@given(a=log_uniform(-2.0, 4.0), ratio=log_uniform(-2.0, 1.0))
def test_incomplete_gamma_matches_mpmath(a, ratio):
    x = a * ratio
    P = mp.gammainc(a, 0, x, regularized=True)
    Q = mp.gammainc(a, x, mp.inf, regularized=True)
    if P > 1e-290:
        assert rel(S.gammainc(a, x), P) < 1e-11
    if Q > 1e-290:
        assert rel(S.gammaincc(a, x), Q) < 1e-11


@profile(100)
@given(a=log_uniform(-2.0, 3.5), b=log_uniform(-2.0, 3.5), x=st.floats(min_value=1e-9, max_value=1 - 1e-9))
def test_incomplete_beta_matches_mpmath(a, b, x):
    I = mp.betainc(a, b, 0, x, regularized=True)
    assume(I > 1e-290)
    got = S.betainc(a, b, x)
    err = rel(got, I)
    if I > 0.5:
        err = min(err, abs(got - float(I)) / 1e-16 * 1e-12)    # near 1, absolute accuracy is what exists
    assert err < 1e-11


@profile(60)
@given(k=log_uniform(-2.0, 4.0), p=st.floats(min_value=1e-6, max_value=1 - 1e-6))
def test_chi2_ppf_inverts_the_mpmath_cdf(k, p):
    x = S.chi2_ppf(p, k)
    assume(x > 0.0)
    if p <= 0.5:
        assert rel(float(mp.gammainc(mp.mpf(k) / 2, 0, mp.mpf(x) / 2, regularized=True)), p) < 1e-10
    else:
        assert rel(float(mp.gammainc(mp.mpf(k) / 2, mp.mpf(x) / 2, mp.inf, regularized=True)), 1 - p) < 1e-10


@profile(40)
@given(a=log_uniform(-2.0, 5.0), df=log_uniform(-1.0, 12.0), seed=st.integers(0, 2**32 - 1))
def test_vectorised_special_functions_agree_with_scalar(a, df, seed):
    rng = np.random.default_rng(seed)
    xs = a * 10.0 ** rng.uniform(-3, 1, size=40)
    for x, p, q in zip(xs, S.gammainc_vec(a, xs), S.gammaincc_vec(a, xs)):
        sp, sq = S.gammainc(a, x), S.gammaincc(a, x)
        assert abs(p - sp) <= 1e-13 * max(sp, 1e-300) or abs(p - sp) < 1e-300
        assert abs(q - sq) <= 1e-13 * max(sq, 1e-300) or abs(q - sq) < 1e-300
    ts = rng.choice([-1, 1], size=40) * 10.0 ** rng.uniform(-8, 5, size=40)
    for t, v in zip(ts, S.t_cdf_vec(ts, df)):
        s = S.t_cdf(t, df)
        assert (rel(v, s) if t < 0 else abs(v - s)) < 1e-12
    zs = rng.uniform(-38, 38, size=40)
    assert np.all(S.norm_cdf_vec(zs) == np.array([S.norm_cdf(z) for z in zs]))
    b = float(10.0 ** rng.uniform(-2, 3))
    aa = min(a, 1e3)
    us = rng.uniform(0, 1, size=40)
    for u, v in zip(us, S.betainc_vec(aa, b, us)):
        s = S.betainc(aa, b, u)
        # both are ~3e-12 from mpmath at a = 56, b = 0.024, x = 0.988 (found by the campaign)
        assert abs(v - s) <= 1e-11 * max(s, 1e-300) or abs(v - s) < 1e-300


# ---------------------------------------------------------------------------
# Copulas: closed forms, Frechet bounds, tau, tail dependence, sampling


def _clayton(u, v, th):
    u, v, th = mp.mpf(u), mp.mpf(v), mp.mpf(th)
    return (u ** -th + v ** -th - 1) ** (-1 / th)


def _gumbel(u, v, th):
    u, v, th = mp.mpf(u), mp.mpf(v), mp.mpf(th)
    return mp.exp(-(((-mp.log(u)) ** th + (-mp.log(v)) ** th) ** (1 / th)))


def _frank(u, v, th):
    # 1 + (...) cancels to about e^(-|theta|·min(u, v)) for a large |theta|: carry enough digits
    # (the campaign caught a 40-digit oracle 4e-12 off at theta = 402).
    with mp.workdps(40 + int(abs(th) / 2.3)):
        u, v, th = mp.mpf(u), mp.mpf(v), mp.mpf(th)
        return +(-1 / th * mp.log1p(mp.expm1(-th * u) * mp.expm1(-th * v) / mp.expm1(-th)))     # log1p: theta ~ 1e-300


def _survival(base):
    return lambda u, v, th: mp.mpf(u) + mp.mpf(v) - 1 + base(1 - mp.mpf(u), 1 - mp.mpf(v), th)


ARCHIMEDEAN = {
    "clayton": (lambda th: ClaytonCopula(th), _clayton, log_uniform(-6.0, 2.5)),
    "survival_clayton": (lambda th: ClaytonCopula(th, survival=True), _survival(_clayton), log_uniform(-6.0, 2.5)),
    "gumbel": (lambda th: GumbelCopula(th), _gumbel, st.floats(min_value=1.0, max_value=200.0)),
    "survival_gumbel": (lambda th: GumbelCopula(th, survival=True), _survival(_gumbel),
                        st.floats(min_value=1.0, max_value=200.0)),
    # down to |theta| = 1e-300: 0.8.0 collapsed below about 1e-160 (finding 8, fixed in 0.9.0)
    "frank": (lambda th: FrankCopula(th), _frank,
              st.one_of(st.floats(min_value=1e-6, max_value=500.0), st.floats(min_value=-500.0, max_value=-1e-6),
                        log_uniform(-300.0, -6.0), log_uniform(-300.0, -6.0).map(lambda t: -t))),
}
UNIT = st.floats(min_value=1e-12, max_value=1 - 1e-12)


@profile(150)
@given(data=st.data(), family=st.sampled_from(sorted(ARCHIMEDEAN)), u=UNIT, v=UNIT)
def test_archimedean_cdf_matches_closed_form(data, family, u, v):
    make, closed, thetas = ARCHIMEDEAN[family]
    th = data.draw(thetas, label="theta")
    exact = closed(u, v, th)
    got = make(th).cdf(u, v)
    # relatively accurate, or within a few ulps of the copula's scale (the survival and
    # negative-Frank forms subtract numbers near u; their corner values are ulp-level).
    assert abs(got - float(exact)) <= 1e-11 * abs(float(exact)) + 4e-16


ALL_COPULAS = st.one_of(
    st.builds(ClaytonCopula, log_uniform(-6.0, 3.0)),
    st.builds(lambda t: ClaytonCopula(t, survival=True), log_uniform(-6.0, 3.0)),
    st.builds(GumbelCopula, st.floats(min_value=1.0, max_value=1e3)),
    st.builds(lambda t: GumbelCopula(t, survival=True), st.floats(min_value=1.0, max_value=1e3)),
    st.builds(FrankCopula, st.floats(min_value=1e-6, max_value=500.0)),
    st.builds(FrankCopula, st.floats(min_value=-500.0, max_value=-1e-6)),
    st.builds(GaussianCopula, st.floats(min_value=-0.999, max_value=0.999)),
    st.builds(StudentTCopula, st.floats(min_value=-0.999, max_value=0.999), log_uniform(-0.5, 3.0)),
)


@profile(120)
@given(cop=ALL_COPULAS, u1=UNIT, u2=UNIT, v1=UNIT, v2=UNIT)
def test_copula_cdf_bounds_monotone_and_2_increasing(cop, u1, u2, v1, v2):
    u1, u2 = sorted((u1, u2))
    v1, v2 = sorted((v1, v2))
    tol = 1e-13
    for u, v in ((u1, v1), (u1, v2), (u2, v1), (u2, v2)):
        c = cop.cdf(u, v)
        assert max(u + v - 1.0, 0.0) - tol <= c <= min(u, v) + tol
    assert cop.cdf(u2, v1) >= cop.cdf(u1, v1) - tol
    assert cop.cdf(u1, v2) >= cop.cdf(u1, v1) - tol
    volume = cop.cdf(u2, v2) - cop.cdf(u1, v2) - cop.cdf(u2, v1) + cop.cdf(u1, v1)
    assert volume >= -4 * tol


# 0.8.0 machine testing, finding 9 (fixed in 0.9.0): GaussianCopula.cdf fell below the Frechet lower bound.
def test_gaussian_cdf_respects_frechet_lower_bound_exactly():
    cop = GaussianCopula(-0.99)
    for u, v in ((0.5, 0.9), (0.3, 0.99), (0.9, 0.9), (0.7, 0.7)):
        assert cop.cdf(u, v) >= u + v - 1.0


def _frank_tau(th):
    th = mp.mpf(th)
    if th < 0:
        return -_frank_tau(-th)
    debye = mp.quad(lambda t: t / mp.expm1(t), [0, th]) / th
    return 1 - 4 / th * (1 - debye)


@profile(60)
@given(th=st.one_of(log_uniform(-7.0, math.log10(499.0)), log_uniform(-7.0, math.log10(499.0)).map(lambda t: -t)))
def test_frank_kendall_tau_matches_debye_integral(th):
    assert abs(FrankCopula(th).kendall_tau() - float(_frank_tau(th))) < 1e-11


@profile(60)
@given(rho=st.floats(min_value=-0.999, max_value=0.999), df=log_uniform(-3.0, 4.0))
def test_t_copula_tail_dependence_matches_mpmath(rho, df):
    x = -mp.sqrt((mp.mpf(df) + 1) * (1 - mp.mpf(rho)) / (1 + mp.mpf(rho)))
    lam = 2 * _t_tail_mp(x, df + 1)
    td = StudentTCopula(rho, df=df).tail_dependence()
    assert td.lower == td.upper
    assert abs(td.upper - float(lam)) < 1e-11 * float(lam) + 1e-300


@profile(30)
@given(th=st.floats(min_value=1.0, max_value=20.0))
def test_archimedean_tail_dependence_agrees_with_cdf_limit(th):
    q = 1e-9
    c = ClaytonCopula(th)
    assert abs(c.cdf(q, q) / q - c.tail_dependence().lower) < 1e-3
    g = GumbelCopula(1.0 + th / 4)
    assert abs(g.joint_exceedance(1 - 1e-7) / 1e-7 - g.tail_dependence().upper) < 1e-3
    assert ClaytonCopula(th).kendall_tau() == pytest.approx(th / (th + 2), abs=1e-15)
    assert GumbelCopula(1 + th).kendall_tau() == pytest.approx(1 - 1 / (1 + th), abs=1e-15)


def _je_exact(family, th, q):
    q = mp.mpf(q)
    C = {"clayton": _clayton, "gumbel": _gumbel, "frank": _frank,
         "survival_clayton": _survival(_clayton), "survival_gumbel": _survival(_gumbel)}[family](q, q, th)
    return 1 - 2 * q + C


@profile(80)
@given(data=st.data(), family=st.sampled_from(sorted(ARCHIMEDEAN)),
       q=st.sampled_from([0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]))
def test_joint_exceedance_matches_high_precision_at_practical_levels(data, family, q):
    make, _, thetas = ARCHIMEDEAN[family]
    th = data.draw(thetas, label="theta")
    exact = _je_exact(family, th, q)
    got = make(th).joint_exceedance(q)
    assert 0.0 <= got <= 1.0 - q + 1e-15
    assert abs(got - float(exact)) <= 1e-6 * float(exact) + 1e-15


# 0.8.0 machine testing, finding 7 (fixed in 0.9.0): joint_exceedance cancelled to noise at high levels.
@pytest.mark.parametrize("cop", [GumbelCopula(1.0, survival=True), GaussianCopula(0.0), FrankCopula(1e-6),
                                 ClaytonCopula(1e-6), ClaytonCopula(1e-6, survival=True)], ids=repr)
def test_joint_exceedance_near_independence_at_high_levels(cop):
    q = 1 - 1e-9
    assert cop.joint_exceedance(q) == pytest.approx((1 - q) ** 2, rel=1e-3, abs=0.0)


SAMPLED = [
    ClaytonCopula(1e-6), ClaytonCopula(0.5), ClaytonCopula(2.0, survival=True), ClaytonCopula(15.0),
    ClaytonCopula(80.0), ClaytonCopula(1e6), ClaytonCopula(1.5, dim=3),
    GumbelCopula(1.0), GumbelCopula(1.5), GumbelCopula(3.0, survival=True), GumbelCopula(50.0), GumbelCopula(2.0, dim=4),
    FrankCopula(-500.0), FrankCopula(-5.0), FrankCopula(1e-8), FrankCopula(5.736), FrankCopula(37.0), FrankCopula(500.0),
    FrankCopula(10.0, dim=3), FrankCopula(100.0, dim=3),
    GaussianCopula(-0.999999), GaussianCopula(0.0), GaussianCopula(0.7), GaussianCopula(1.0),
    StudentTCopula(0.5, df=1e-6), StudentTCopula(0.0, df=0.19), StudentTCopula(0.5, df=0.2), StudentTCopula(-0.3, df=4.0),
    StudentTCopula(0.9, df=1e300),
]


@pytest.mark.parametrize("cop", SAMPLED, ids=repr)
def test_copula_samples_uniform_margins_and_kendall_tau(cop):
    u = cop.sample(20_000, seed=20261121)
    assert u.shape == (20_000, cop.dim)
    assert np.all(np.isfinite(u)) and u.min() > 0.0 and u.max() < 1.0
    for j in range(cop.dim):
        assert stats.kstest(u[:, j], "uniform").pvalue > 1e-6
    tau = cop.kendall_tau() if not isinstance(cop, (GaussianCopula, StudentTCopula)) else cop.kendall_tau(0, 1)
    sample_tau = stats.kendalltau(u[:4000, 0], u[:4000, 1]).statistic
    # heavy-tailed t copulas with df < 1 have a sample tau several times noisier than 1/sqrt(n)
    tolerance = 0.08 if isinstance(cop, StudentTCopula) and cop.df < 1 else 0.04
    assert abs(sample_tau - tau) < tolerance


@profile(40)
@given(cop=st.one_of(
    st.builds(ClaytonCopula, st.one_of(log_uniform(-12.0, 300.0))),
    st.builds(GumbelCopula, st.one_of(st.just(1.0), log_uniform(-12.0, 300.0).map(lambda t: 1.0 + t))),
    st.builds(FrankCopula, st.floats(min_value=1e-6, max_value=500.0)),
    st.builds(StudentTCopula, st.floats(min_value=-1.0, max_value=1.0), log_uniform(-300.0, 300.0)),
), seed=st.integers(0, 2**32 - 1))
def test_extreme_parameters_sample_inside_the_open_unit_square(cop, seed):
    u = cop.sample(2000, seed=seed)
    assert np.all(np.isfinite(u)) and u.min() > 0.0 and u.max() < 1.0


# ---------------------------------------------------------------------------
# nearest_correlation, allocate


def _higham(a, iterations=20_000):
    y, ds = a.copy(), np.zeros_like(a)
    for _ in range(iterations):
        r = y - ds
        w, vec = np.linalg.eigh(r)
        x = (vec * np.clip(w, 0, None)) @ vec.T
        ds = x - r
        nxt = x.copy()
        np.fill_diagonal(nxt, 1.0)
        if np.linalg.norm(nxt - y) < 1e-15:
            return nxt
        y = nxt
    return y


@profile(40)
@given(k=st.integers(2, 7), seed=st.integers(0, 2**32 - 1), valid=st.booleans())
def test_nearest_correlation_is_a_correlation_matrix_and_nearest(k, seed, valid):
    rng = np.random.default_rng(seed)
    if valid:
        a = np.corrcoef(rng.normal(size=(k, k + 3)))
    else:
        a = rng.uniform(-1, 1, size=(k, k))
        a = (a + a.T) / 2
        np.fill_diagonal(a, 1.0)
    out = nearest_correlation(a)
    assert np.array_equal(out, out.T)
    assert np.all(np.diag(out) == 1.0)
    assert np.abs(out).max() <= 1.0
    assert np.linalg.eigvalsh(out).min() >= -1e-12
    if np.linalg.eigvalsh(a).min() >= -1e-12:
        assert np.allclose(out, a, atol=1e-15, rtol=0)
    else:
        ref = _higham(a)
        assert np.linalg.norm(out - a) <= np.linalg.norm(ref - a) * (1 + 1e-8) + 1e-12


def _expected_shortfall(x, level):
    xs = np.sort(x)[::-1]
    m = x.size * (1 - level)
    whole = int(math.floor(m + 1e-9))
    tail = xs[:whole].sum() + (m - whole) * (xs[whole] if whole < x.size else 0.0)
    return tail / m


@profile(60)
@given(n=st.integers(2, 3000), k=st.integers(1, 6), seed=st.integers(0, 2**32 - 1), ties=st.booleans(),
       level=st.one_of(st.sampled_from([0.5, 0.9, 0.95, 0.99, 0.995]), st.floats(min_value=0.01, max_value=0.999)))
def test_allocation_adds_up_and_total_is_expected_shortfall(n, k, seed, ties, level):
    rng = np.random.default_rng(seed)
    losses = rng.lognormal(size=(n, k)) * rng.uniform(0.1, 100, size=k)
    if ties:
        losses = np.round(losses)
    for method in ("euler", "covariance", "standalone"):
        for measure in ("tvar", "var"):
            a = allocate(losses, level=level, measure=measure, method=method)
            assert sum(a.contributions.values()) == pytest.approx(a.total, rel=1e-12, abs=1e-12)
    a = allocate(losses, level=level)
    assert a.total == pytest.approx(_expected_shortfall(losses.sum(axis=1), level), rel=1e-12)


# ---------------------------------------------------------------------------
# Monte Carlo: Wald identities, stress and reverse stress


FREQUENCIES = st.one_of(
    st.builds(mc.Poisson, st.floats(min_value=1e-3, max_value=200.0)),
    st.builds(mc.NegativeBinomial, st.floats(min_value=0.5, max_value=50.0), st.floats(min_value=0.05, max_value=0.95)),
    st.builds(mc.Binomial, st.integers(1, 100), st.floats(min_value=0.01, max_value=0.99)),
)
SEVERITIES = st.one_of(
    st.builds(mc.LogNormal, st.floats(min_value=-2.0, max_value=10.0), st.floats(min_value=0.05, max_value=1.5)),
    st.builds(mc.Gamma, st.floats(min_value=0.2, max_value=20.0), st.floats(min_value=0.1, max_value=1e4)),
    st.builds(mc.Exponential, st.floats(min_value=0.01, max_value=1e4)),
)


@profile(60)
@given(freq=FREQUENCIES, sev=SEVERITIES)
def test_compound_moments_are_the_wald_identities(freq, sev):
    f, s, m = freq.moments(), sev.moments(), mc.Compound(freq, sev).moments()
    assert m.mean == pytest.approx(f.mean * s.mean, rel=1e-14)
    assert m.variance == pytest.approx(f.mean * s.variance + f.variance * s.mean ** 2, rel=1e-14)


@pytest.mark.parametrize("freq,sev", [
    (mc.Poisson(0.2), mc.Gamma(2.0, 3.0)), (mc.Poisson(40.0), mc.Exponential(0.5)),
    (mc.NegativeBinomial(5, 0.4), mc.LogNormal(0.0, 1.0)), (mc.Binomial(10, 0.3), mc.Exponential(2.0)),
], ids=repr)
def test_compound_sample_moments_match_wald(freq, sev):
    comp = mc.Compound(freq, sev)
    m = comp.moments()
    for draw in (comp.sample, comp.sample_lhs):
        x = draw(400_000, np.random.default_rng(7))
        assert abs(x.mean() - m.mean) < 5 * math.sqrt(m.variance / x.size)
        assert x.var(ddof=1) == pytest.approx(m.variance, rel=0.05)


def _stress_model():
    m = mc.Model(claims=mc.LogNormal(10.0, 0.8), inflation=mc.Normal(0.03, 0.01), count=mc.Poisson(2.0))

    @m.formula
    def loss(claims, inflation, count):
        return claims * (1 + inflation) + 1000.0 * count

    return m


@profile(15)
@given(seed=st.integers(1, 2**31 - 1), trials=st.integers(50, 5000))
def test_stress_shocks_do_not_leak_and_redraws_keep_ranks(seed, trials):
    m = _stress_model()
    run = m.run(trials, seed=seed)
    scenarios = {
        "pin": {"inflation": 0.08},
        "in place": {"claims": lambda x: x.__imul__(1.2)},
        "redraw": {"claims": mc.LogNormal(10.2, 0.9)},
        "redraw count": {"count": mc.Poisson(5.0)},
    }
    stress = m.stress(scenarios, trials=trials, seed=seed)
    assert np.array_equal(stress.base.values, run.values)
    for name in run.inputs:
        assert np.array_equal(stress.base.inputs[name], run.inputs[name])
    s = stress.scenarios
    assert np.array_equal(s["pin"].inputs["claims"], run.inputs["claims"])
    assert np.all(s["pin"].inputs["inflation"] == 0.08)
    assert np.allclose(s["in place"].inputs["claims"], 1.2 * run.inputs["claims"])
    base_claims, new_claims = run.inputs["claims"], s["redraw"].inputs["claims"]
    assert np.array_equal(np.argsort(base_claims, kind="stable"), np.argsort(new_claims, kind="stable"))
    counts, new_counts = run.inputs["count"], s["redraw count"].inputs["count"]
    levels = np.unique(counts)
    for lo, hi in zip(levels, levels[1:]):
        assert new_counts[counts == lo].max() <= new_counts[counts == hi].min()
    alone = m.stress({"redraw": scenarios["redraw"]}, trials=trials, seed=seed)
    assert np.array_equal(alone.scenarios["redraw"].values, s["redraw"].values)


# 0.8.0 machine testing, finding 3 (fixed in 0.9.0): a formula's in-place edit leaked into every scenario.
def test_stress_formula_that_mutates_an_input_does_not_leak_into_scenarios():
    m = mc.Model(claims=mc.LogNormal(10.0, 0.8), deductible=mc.Uniform(0.0, 1000.0))

    @m.formula
    def net(claims, deductible):
        claims -= deductible
        return np.maximum(claims, 0.0)

    stress = m.stress({"no-op": {"deductible": lambda x: x}}, trials=5000, seed=3)
    assert stress.delta("no-op", "mean") == 0.0


@profile(30)
@given(seed=st.integers(1, 2**31 - 1), trials=st.integers(20, 20_000), level=st.floats(min_value=0.01, max_value=0.9999),
       integer_output=st.booleans())
def test_reverse_stress_tail_matches_level_and_tvar(seed, trials, level, integer_output):
    m = mc.Model(a=mc.Poisson(3.0) if integer_output else mc.LogNormal(0.0, 1.0), b=mc.Normal(0.0, 1.0))
    m.formula(lambda a, b: a * 1.0)
    result = m.run(trials, seed=seed)
    rs = result.reverse_stress(level)
    assert rs.output_mean == pytest.approx(result.tvar(level), rel=1e-9)
    if not integer_output:          # continuous: no ties, so the tail is exactly ceil(n(1 - level)) trials
        assert rs.tail_trials == math.ceil(trials * (1 - level) - 1e-9)
    low = result.reverse_stress(level, tail="lower")
    mirrored = mc.Result(values=-result.values, inputs={}, trials=trials, seed=None)
    assert low.output_mean == pytest.approx(-mirrored.tvar(level), rel=1e-9, abs=1e-12)
    threshold = float(np.quantile(result.values, 0.5))
    rt = result.reverse_stress(threshold=threshold)
    assert rt.tail_trials == int((result.values >= threshold).sum())


# ---------------------------------------------------------------------------
# Closed forms: options, bonds, rates, annuities, life, reserving, credit


def _bs_mp(S_, K, T, r, s, q, kind):
    S_, K, T, r, s, q = map(mp.mpf, (S_, K, T, r, s, q))
    d1 = (mp.log(S_ / K) + (r - q + s * s / 2) * T) / (s * mp.sqrt(T))
    d2 = d1 - s * mp.sqrt(T)
    if kind == "call":
        return S_ * mp.e ** (-q * T) * mp.ncdf(d1) - K * mp.e ** (-r * T) * mp.ncdf(d2)
    return K * mp.e ** (-r * T) * mp.ncdf(-d2) - S_ * mp.e ** (-q * T) * mp.ncdf(-d1)


OPTION = st.tuples(log_uniform(-2.0, 4.0), log_uniform(-1.0, 1.0), log_uniform(-3.0, 1.5),
                   st.floats(min_value=-0.05, max_value=0.2), log_uniform(-3.0, 0.5),
                   st.floats(min_value=0.0, max_value=0.1))


@profile(80)
@given(OPTION)
def test_black_scholes_matches_mpmath_and_parity(opt):
    S_, moneyness, T, r, s, q = opt
    K = S_ * moneyness
    scale = max(S_, K)
    for kind in ("call", "put"):
        exact = _bs_mp(S_, K, T, r, s, q, kind)
        got = quant.black_scholes(S_, K, T, r, s, q, kind)
        assert abs(got - float(exact)) <= 1e-15 * scale + 1e-11 * float(exact)
    c, p = quant.black_scholes(S_, K, T, r, s, q, "call"), quant.black_scholes(S_, K, T, r, s, q, "put")
    assert abs(c - p - (S_ * math.exp(-q * T) - K * math.exp(-r * T))) <= 2e-15 * scale


@profile(40)
@given(OPTION, st.sampled_from(["call", "put"]))
def test_greeks_match_mpmath_derivatives(opt, kind):
    S_, moneyness, T, r, s, q = opt
    K = S_ * moneyness
    g = quant.greeks(S_, K, T, r, s, q, kind)
    checks = (
        (g.delta, mp.diff(lambda x: _bs_mp(x, K, T, r, s, q, kind), S_), 1.0),
        (g.gamma, mp.diff(lambda x: _bs_mp(x, K, T, r, s, q, kind), S_, 2), 1.0 / S_),
        (g.vega, mp.diff(lambda x: _bs_mp(S_, K, T, r, x, q, kind), s) / 100, S_ / 100),
        (g.theta, -mp.diff(lambda x: _bs_mp(S_, K, x, r, s, q, kind), T) / 365, max(S_, K) / 365),
        (g.rho, mp.diff(lambda x: _bs_mp(S_, K, T, x, s, q, kind), r) / 100, K * T / 100),
    )
    for got, exact, scale in checks:
        assert abs(got - float(exact)) <= 1e-10 * scale


@profile(60)
@given(OPTION)
def test_implied_vol_recovers_sigma(opt):
    S_, moneyness, T, r, s, q = opt
    K = S_ * moneyness
    price = quant.black_scholes(S_, K, T, r, s, q, "call")
    vega = quant.greeks(S_, K, T, r, s, q).vega * 100
    assume(vega > 1e-6 * S_ and price > 1e-10 * S_)
    iv = quant.implied_vol(price, S_, K, T, r, q, "call")
    assert quant.black_scholes(S_, K, T, r, iv, q, "call") == pytest.approx(price, rel=1e-8, abs=1e-12)


@profile(80)
@given(face=log_uniform(0.0, 6.0), coupon=st.floats(min_value=0.0, max_value=0.15),
       frequency=st.sampled_from([1, 2, 4, 12]), periods=st.integers(1, 600), ytm=st.floats(min_value=-0.02, max_value=0.3))
def test_bond_price_is_the_discounted_cash_flows_and_ytm_inverts_it(face, coupon, frequency, periods, ytm):
    maturity = periods / frequency
    y = mp.mpf(ytm) / frequency
    exact = sum(face * coupon / frequency / (1 + y) ** k for k in range(1, periods + 1)) + face / (1 + y) ** periods
    price = rates.bond_price(face, coupon, maturity, ytm, frequency)
    assert rel(price, exact) < 1e-12
    bond = rates.Bond(face=face, coupon=coupon, maturity=maturity, frequency=frequency)
    assert bond.yield_to_maturity(price) == pytest.approx(ytm, abs=1e-12)


# 0.8.0 machine testing, finding 12 (fixed in 0.9.0): yield_to_maturity overflowed for annual bonds past 154 years.
def test_bond_ytm_solves_long_annual_bonds():
    bond = rates.Bond(face=100.0, coupon=0.04, maturity=160.0, frequency=1)
    assert bond.price(bond.yield_to_maturity(90.0)) == pytest.approx(90.0, rel=1e-10)


COMPOUNDINGS = st.sampled_from(["continuous", "annual", "semiannual", "quarterly", "monthly", 1, 2, 4, 12, 365])


@profile(80)
@given(rate=st.floats(min_value=-0.5, max_value=1.0), a=COMPOUNDINGS, b=COMPOUNDINGS)
def test_convert_rate_round_trips_and_preserves_one_year_growth(rate, a, b):
    x = rates.convert_rate(rate, a, b)
    assert rates.convert_rate(x, b, a) == pytest.approx(rate, abs=1e-14)
    periods = {"continuous": None, "annual": 1, "semiannual": 2, "quarterly": 4, "monthly": 12}

    def growth(r, c):
        m = periods.get(c, c)
        return mp.e ** mp.mpf(r) if m is None else (1 + mp.mpf(r) / m) ** m

    assert rel(float(growth(x, b)), growth(rate, a)) < 1e-14


@profile(80)
@given(i=st.floats(min_value=-0.05, max_value=0.5), n=st.integers(0, 500))
def test_annuity_certain_identities(i, n):
    v = 1 / (1 + mp.mpf(i))
    immediate = sum(v ** k for k in range(1, n + 1))
    a, due = life.annuity_certain(i, n), life.annuity_certain(i, n, due=True)
    assert abs(a - float(immediate)) <= 1e-13 * max(float(immediate), 1.0)
    assert abs(due - float((1 + mp.mpf(i)) * immediate)) <= 1e-13 * max(float(immediate), 1.0)
    if n >= 1:
        assert due == pytest.approx(1 + life.annuity_certain(i, n - 1), rel=1e-13, abs=1e-13)


SULT = life.LifeTable.sult()


@profile(80)
@given(x=st.integers(20, 125), i=st.floats(min_value=0.0, max_value=0.15), data=st.data())
def test_life_insurance_and_annuity_identities(x, i, data):
    d = i / (1 + i)
    A, a = life.whole_life_insurance(SULT, x, i), life.whole_life_annuity_due(SULT, x, i)
    assert abs(A - (1 - d * a)) < 1e-13
    lx = [SULT.lx(k) for k in range(x, SULT.omega + 1)]
    v = 1 / (1 + mp.mpf(i))
    assert rel(a, sum(v ** k * lx[k] for k in range(len(lx))) / lx[0]) < 1e-13
    n = data.draw(st.integers(1, 130 - x), label="n")
    E, term, pure = (life.endowment_insurance(SULT, x, n, i), life.term_insurance(SULT, x, n, i),
                     life.pure_endowment(SULT, x, n, i))
    assert abs(E - term - pure) < 1e-14
    assert abs(E - (1 - d * life.temporary_annuity_due(SULT, x, n, i))) < 1e-13


@profile(60)
@given(x=st.integers(20, 120), i=st.floats(min_value=0.0, max_value=0.15),
       kind=st.sampled_from(["whole_life", "term", "endowment"]), data=st.data())
def test_net_premium_reserve_starts_at_zero_and_follows_fackler(x, i, kind, data):
    n = None if kind == "whole_life" else data.draw(st.integers(1, 130 - x), label="n")
    last = SULT.omega - x - 1 if kind == "whole_life" else n
    assume(last >= 1)
    t = data.draw(st.integers(0, last - 1), label="t")
    premium = (life.whole_life_premium(SULT, x, i) if kind == "whole_life" else
               {"term": life.term_premium, "endowment": life.endowment_premium}[kind](SULT, x, n, i))
    assert abs(life.net_premium_reserve(SULT, x, 0, i, kind, n)) < 1e-14
    v_t = life.net_premium_reserve(SULT, x, t, i, kind, n)
    v_next = life.net_premium_reserve(SULT, x, t + 1, i, kind, n)
    q = SULT.qx(x + t)
    assert abs((v_t + premium) * (1 + i) - (q + (1 - q) * v_next)) < 1e-13


@profile(60)
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(2, 15), tail=st.sampled_from([1.0, 1.02, 1.1]))
def test_chain_ladder_matches_hand_calculation(seed, n, tail):
    rng = np.random.default_rng(seed)
    growth = rng.uniform(1.0, 2.0, size=n - 1)
    triangle = []
    for i in range(n):
        row = [float(rng.uniform(100, 1e6))]
        for k in range(n - 1 - i):
            row.append(row[-1] * growth[k] * rng.uniform(1.0, 1.1))
        triangle.append(row + [None] * (n - len(row)))
    factors = []
    for k in range(n - 1):
        rows = [r for r in triangle if r[k + 1] is not None]
        factors.append(sum(r[k + 1] for r in rows) / sum(r[k] for r in rows))
    result = reserving.chain_ladder(triangle, tail=tail)
    assert np.allclose(result.factors, factors, rtol=1e-13, atol=0)
    for i, row in enumerate(triangle):
        latest = row[n - 1 - i]
        ultimate = latest * math.prod(factors[n - 1 - i:]) * tail
        assert result.ultimate[i] == pytest.approx(ultimate, rel=1e-13)
        assert result.reserve[i] == pytest.approx(ultimate - latest, rel=1e-12, abs=1e-9 * ultimate)


@profile(60)
@given(V=log_uniform(1.0, 4.0), leverage=st.floats(min_value=0.1, max_value=1.5), T=st.floats(min_value=0.1, max_value=10.0),
       r=st.floats(min_value=0.0, max_value=0.1), s=st.floats(min_value=0.01, max_value=1.0))
def test_merton_equity_is_a_call_and_pd_is_n_of_minus_d2(V, leverage, T, r, s):
    D = V * leverage
    m = credit.merton(V, D, T, r, s)
    assert m.equity == quant.black_scholes(V, D, T, r, s)
    d2 = (mp.log(mp.mpf(V) / D) + (r - s * s / 2) * T) / (s * mp.sqrt(T))
    exact = mp.ncdf(-d2)
    if exact > 1e-290:      # below, the pd is subnormal and carries only a few digits anyway
        # N(-d2) has relative condition number ~d2^2 in d2, and d2 itself carries ln(V/D), which
        # loses relative digits as V/D -> 1 (campaign: 1.3e-12 at V = 10, D = 9.375, d2 = 15.6)
        assert rel(m.pd, exact) < 1e-12 + 1e-14 * float(d2) ** 2


@profile(60)
@given(level=st.floats(min_value=0.5, max_value=0.99999), pd=log_uniform(-6.0, -0.05),
       rho=st.floats(min_value=0.001, max_value=0.99))
def test_vasicek_loss_quantile_formula(level, pd, rho):
    g = lambda p: mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1)  # noqa: E731
    exact = mp.ncdf((g(pd) + mp.sqrt(rho) * g(level)) / mp.sqrt(1 - mp.mpf(rho)))
    assert rel(credit.vasicek_loss_quantile(level, pd, rho), exact) < 1e-11


@profile(60)
@given(hazard=log_uniform(-4.0, 1.0), recovery=st.floats(min_value=0.0, max_value=0.9),
       maturity=st.floats(min_value=0.25, max_value=30.0), frequency=st.sampled_from([1, 2, 4, 12]))
def test_cds_par_spread_is_the_credit_triangle_at_zero_rates(hazard, recovery, maturity, frequency):
    """docs: with r = 0 the exact legs reproduce h(1 - R) at every hazard and frequency."""
    assert credit.cds_par_spread(hazard, recovery, 0.0, maturity, frequency) == pytest.approx(
        hazard * (1 - recovery), rel=1e-12)


@profile(60)
@given(pd=st.floats(min_value=1e-5, max_value=0.999), lgd=st.floats(min_value=0.0, max_value=1.0),
       maturity=st.floats(min_value=1.0, max_value=5.0))
def test_basel_irb_corporate_matches_the_accord_formula(pd, lgd, maturity):
    g = lambda p: mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1)  # noqa: E731
    w = (1 - mp.e ** (-50 * mp.mpf(pd))) / (1 - mp.e ** (-50))
    R = mp.mpf("0.12") * w + mp.mpf("0.24") * (1 - w)
    b = (mp.mpf("0.11852") - mp.mpf("0.05478") * mp.log(pd)) ** 2
    K = lgd * (mp.ncdf((g(pd) + mp.sqrt(R) * g(mp.mpf("0.999"))) / mp.sqrt(1 - R)) - pd) \
        * (1 + (maturity - mp.mpf("2.5")) * b) / (1 - mp.mpf("1.5") * b)
    got = credit.basel_irb_capital(pd, lgd, 1.0, maturity).capital_requirement
    assert abs(got - float(K)) <= 1e-12 * max(abs(float(K)), 1e-3)


# ==============================================================================================================
# Audit ledger properties (workstream 3)
# (from test_audit_props.py)
# ==============================================================================================================

"""Property tests for riskpy.audit (the hash-chained ledger): round trip, tamper evidence, continuation,
concurrency. Deterministic under the fixed Hypothesis profile below; strict xfails pin defects found by the
0.8.0 machine-testing pass (remove the marker when the fix lands).
"""

import hashlib
import io
import json
import math
import multiprocessing as mproc
import os
import threading

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from riskpy import audit, tools

AUDIT_PROFILE = settings(derandomize=True, max_examples=25, deadline=None, database=None,
                   suppress_health_check=list(HealthCheck))

EL = {"pd": 0.02, "lgd": 0.45, "ead": 1000.0}
AUDIT_BS = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}
AGG = {"expected_frequency": 2.0, "severity_mean": 1000.0, "severity_sd": 500.0, "trials": 1000, "seed": 7}
CALLS = [
    ("expected_loss", EL), ("black_scholes", AUDIT_BS), ("annuity_certain", {"i": 0.05, "n": 10}),
    ("bond_analytics", {"face": 100.0, "coupon": 0.05, "maturity": 5.0, "ytm": 0.04}),
    ("capital_aggregate", {"standalone": [100.0, 60.0], "correlation": [[1.0, 0.25], [0.25, 1.0]]}),
    ("aggregate_loss", AGG),
    ("scenario_table", {"tool": "black_scholes", "arguments": AUDIT_BS, "scenarios": {"vol 30%": {"sigma": 0.3}}}),
    ("sensitivity", {"tool": "black_scholes", "arguments": AUDIT_BS, "parameter": "S"}),
]
BAD_VALUES = [math.nan, math.inf, -0.0, 10 ** 30, "100", None, True, [1.0, [2.0]], b"\x00\xff", {"$riskpy": "float"}]

json_leaf = st.one_of(st.none(), st.booleans(), st.integers(-(10 ** 30), 10 ** 30),
                      st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=10))
json_value = st.recursive(json_leaf, lambda c: st.one_of(st.lists(c, max_size=4),
                                                         st.dictionaries(st.text(max_size=6), c, max_size=4)),
                          max_leaves=10)
call = st.tuples(st.sampled_from(range(len(CALLS))), st.one_of(st.none(), st.sampled_from(BAD_VALUES)))
op = st.one_of(call.map(lambda c: ("call",) + c),
               st.tuples(st.just("append"), st.text(max_size=8), json_value, json_value))


def _run(ops):
    n = 0
    for o in ops:
        if o[0] == "call":
            _, k, bad = o
            name, args = CALLS[k]
            args = dict(args)
            if bad is not None:
                args[sorted(args)[0]] = bad
            tools.call(name, args, errors="return")
        else:
            audit.active().append("x_" + o[1], o[2], result=o[3])
        n += 1
    return n


@AUDIT_PROFILE
@given(st.lists(op, min_size=1, max_size=6))
def test_any_sequence_of_calls_verifies_loads_and_replays_without_drift(tmp_path_factory, ops):
    path = str(tmp_path_factory.mktemp("rt") / "run.jsonl")
    with audit.record(path, note="props ✓") as ledger:
        n = _run(ops)
    chain = audit.verify_chain(path)
    assert chain.ok and chain.entries == n and chain.head == ledger.head
    assert audit.verify_chain(path, expected_head=ledger.head, expected_entries=n).ok
    assert audit.load(path) == ledger.entries
    rep = audit.replay(path)
    assert rep.chain.ok and not rep.drifted
    assert all("no such tool" in s["reason"] for s in rep.skipped)       # only the direct appends
    assert rep.matched + len(rep.skipped) == n
    assert isinstance(audit.report(path, check=True), str)


def _small_ledger(path):
    with audit.record(path, note="tamper é") as led:
        tools.call("black_scholes", AUDIT_BS)
        tools.call("black_scholes", dict(AUDIT_BS, sigma=math.nan), errors="return")
        led.append("x_custom", {"b": b"\x00\xff", "neg0": -0.0, "big": 10 ** 30, "s": "😀"},
                   result={"v": [1.5e-7, 2.0, -3, True, None]})
    return open(path, "rb").read(), led.head, len(led)


def _parsed(data: bytes):
    out = []
    for raw in data.split(b"\n"):
        if raw.strip():
            entry = audit._parse_line(raw)
            if not isinstance(entry, dict):
                return None
            try:
                out.append(json.dumps(entry, sort_keys=True, ensure_ascii=True, allow_nan=False))
            except ValueError:
                return None
    return out


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("small") / "base.jsonl")
    return (path,) + _small_ledger(path)


@AUDIT_PROFILE
@given(data=st.data())
def test_every_value_changing_byte_edit_is_detected(small, tmp_path_factory, data):
    path, original, head, n = small
    i = data.draw(st.integers(0, len(original) - 1))
    how = data.draw(st.sampled_from(["flip", "replace", "delete", "insert", "truncate"]))
    byte = data.draw(st.integers(0, 255))
    edited = {"flip": original[:i] + bytes([original[i] ^ (1 << (byte % 8))]) + original[i + 1:],
              "replace": original[:i] + bytes([byte]) + original[i + 1:],
              "delete": original[:i] + original[i + 1:],
              "insert": original[:i] + bytes([byte]) + original[i:],
              "truncate": original[:i]}[how]
    target = str(tmp_path_factory.mktemp("edit") / "e.jsonl")
    open(target, "wb").write(edited)
    if _parsed(edited) == _parsed(original):
        return                                   # spacing, escape spelling: the values are unchanged
    assert not audit.verify_chain(target, expected_head=head, expected_entries=n).ok


def test_line_edits_inside_the_chain_are_detected_without_the_head(small, tmp_path):
    path, original, head, n = small
    lines = original.split(b"\n")[:-1]
    edits = []
    for k in range(len(lines)):
        edits.append([l for j, l in enumerate(lines) if j != k][:] if k < len(lines) - 1 else None)
        edits.append(lines[:k + 1] + [lines[k]] + lines[k + 1:])
        if k + 1 < len(lines):
            sw = list(lines)
            sw[k], sw[k + 1] = sw[k + 1], sw[k]
            edits.append(sw)
    for e in edits:
        if e is None:
            continue
        p = tmp_path / "x.jsonl"
        p.write_bytes(b"".join(l + b"\n" for l in e))
        assert not audit.verify_chain(str(p)).ok


def test_cutting_whole_entries_off_the_end_needs_the_expected_head(small, tmp_path):
    path, original, head, n = small
    lines = original.split(b"\n")[:-1]
    for keep in range(len(lines)):
        p = tmp_path / "cut.jsonl"
        p.write_bytes(b"".join(l + b"\n" for l in lines[:keep]))
        assert audit.verify_chain(str(p)).ok                    # documented: a valid, shorter chain
        assert not audit.verify_chain(str(p), expected_head=head).ok
        assert not audit.verify_chain(str(p), expected_entries=n).ok


piece = st.one_of(st.tuples(st.just("entry"), st.integers(1, 2)), st.tuples(st.just("torn"), st.floats(0.1, 0.9)),
                  st.tuples(st.just("foreign"), st.sampled_from([b"hello", b"[1]", b"\xff\xfe"])),
                  st.tuples(st.just("blank"), st.sampled_from([b"", b"  "])))


@AUDIT_PROFILE
@given(st.lists(piece, max_size=5), st.integers(1, 3), st.booleans())
def test_a_damaged_ledger_is_continued_on_new_lines_and_the_damage_stays_visible(tmp_path_factory, pieces, k, strip):
    p = str(tmp_path_factory.mktemp("cont") / "c.jsonl")
    writer = audit.Ledger(p)
    damaged = False
    good_seq = 0
    for kind, arg in pieces:
        if kind == "entry":
            for _ in range(arg):
                good_seq = writer.append("expected_loss", EL, result=9.0)["seq"]
        else:
            line = (json.dumps({"seq": good_seq + 1, "digest": "c" * 64}) + "\n").encode()
            data = line[:max(1, int(len(line) * arg))].rstrip(b"\n") if kind == "torn" else arg + b"\n"
            with open(p, "ab") as fh:
                fh.write(data)
            damaged |= bool(data.strip())
    if strip and os.path.exists(p):
        with open(p, "rb") as fh:
            body = fh.read()
        with open(p, "wb") as fh:
            fh.write(body.rstrip(b"\n"))
    before = open(p, "rb").read() if os.path.exists(p) else b""
    new = [audit.Ledger(p).append("expected_loss", EL, result=9.0) for _ in range(k)]
    after = open(p, "rb").read()
    assert after.startswith(before)
    assert len([l for l in after[len(before):].split(b"\n") if l.strip()]) == k
    assert [e["seq"] for e in new] == list(range(good_seq + 1, good_seq + k + 1))
    assert audit.verify_chain(p).ok is (not damaged)


@AUDIT_PROFILE
@given(st.binary(max_size=300), st.integers(1, 9))
def test_reading_lines_backwards_matches_a_forward_split(data, block):
    old = audit._BLOCK
    audit._BLOCK = block
    try:
        got = list(audit._lines_backwards(io.BytesIO(data), len(data)))
    finally:
        audit._BLOCK = old
    assert got == [l for l in reversed(data.split(b"\n")) if l.strip()]


exotic = st.sampled_from([math.nan, math.inf, -math.inf, -0.0, 10 ** 700, -(2 ** 3000), b"\x00", "\ud800",
                          {"$riskpy": "float", "value": "nan"}, {1: "a"}, frozenset({1, 2}), {2.5}])
anyv = st.recursive(st.one_of(json_leaf, exotic), lambda c: st.one_of(
    st.lists(c, max_size=3), st.dictionaries(st.text(max_size=4), c, max_size=3)), max_leaves=8)


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1, a) == math.copysign(1, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


@AUDIT_PROFILE
@given(anyv)
def test_arguments_are_stored_exactly(value):
    stored = audit._encode(value)
    text = json.dumps(stored, ensure_ascii=True, allow_nan=False)        # always strict JSON
    assert _same(audit._decode(json.loads(text)), value)


def test_threads_recording_through_one_ledger_never_corrupt_the_chain(tmp_path):
    path = str(tmp_path / "threads.jsonl")
    barrier = threading.Barrier(8)

    def work(t):
        barrier.wait()
        for j in range(20):
            tools.call("expected_loss", dict(EL, ead=1000.0 + 100 * t + j))

    with audit.record(path):
        ts = [threading.Thread(target=work, args=(t,)) for t in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    chain = audit.verify_chain(path)
    assert chain.ok and chain.entries == 160
    rep = audit.replay(path)
    assert rep.matched == 160 and not rep.drifted


def _proc(path, m, start_at):
    import time
    from riskpy import audit as a, tools as t
    a.start(path)
    while time.time() < start_at:
        pass
    for j in range(m):
        t.call("expected_loss", {"pd": 0.02, "lgd": 0.45, "ead": 1000.0 + j})
    a.stop()


def test_concurrent_processes_fork_the_chain_visibly_never_silently(tmp_path):
    import time
    path = str(tmp_path / "procs.jsonl")
    ctx = mproc.get_context("spawn")
    start_at = time.time() + 2.0
    ps = [ctx.Process(target=_proc, args=(path, 15, start_at)) for _ in range(4)]
    [p.start() for p in ps]
    [p.join(60) for p in ps]
    chain = audit.verify_chain(path)
    lines = [l for l in open(path, "rb").read().split(b"\n") if l.strip()]
    assert len(lines) == 60                                   # every entry landed on a line of its own
    assert all(isinstance(audit._parse_line(l), dict) for l in lines)
    assert chain.ok is (chain.entries == 60 and all(json.loads(l)["seq"] == k + 1 for k, l in enumerate(lines)))


# ---------------------------------------------------------------------------- defects found in 0.8.0

# 0.8.0 machine testing, finding 4 (fixed in 0.9.0): replay crashed on an int beyond float range.
def test_replay_survives_a_recorded_integer_beyond_float_range():
    with audit.record() as led:
        out = tools.call("scenario_table", {"tool": "black_scholes", "arguments": AUDIT_BS,
                                            "scenarios": {"huge": {"S": 10 ** 400}}})
    assert "error" in out["scenarios"]["huge"]
    assert audit.verify_chain(led).ok
    assert audit.replay(led).ok


# 0.8.0 machine testing, finding 5 (fixed in 0.9.0): audit.record() was process-global.
def test_record_blocks_in_two_threads_keep_their_own_calls():
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    got = {}

    def a():
        with audit.record() as la:
            a_in.set(); b_in.wait()
            tools.call("expected_loss", dict(EL, ead=1.0))
        got["A"] = la
        a_out.set()

    def b():
        a_in.wait()
        with audit.record() as lb:
            b_in.set(); a_out.wait()
            tools.call("expected_loss", dict(EL, ead=2.0))
        got["B"] = lb

    ts = [threading.Thread(target=a), threading.Thread(target=b)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    stale = audit.active()
    audit.stop()
    assert [e["arguments"]["ead"] for e in got["A"].entries] == [1.0]
    assert [e["arguments"]["ead"] for e in got["B"].entries] == [2.0]
    assert stale is None


# 0.8.0 machine testing, finding 2 (fixed in 0.9.0): a seed-0 scenario ran and replay reported drift.
def test_replay_skips_meta_tool_calls_whose_inner_runs_draw_a_fresh_seed():
    with audit.record() as led:
        tools.call("scenario_table", {"tool": "aggregate_loss", "arguments": AGG, "scenarios": {"fresh": {"seed": 0}}},
                   errors="return")
    assert not audit.replay(led).drifted


# 0.8.0 machine testing, finding 16 (fixed in 0.9.0): false drift for an over-long tool name.
def test_replay_of_a_call_refused_for_an_over_long_tool_name_matches():
    with audit.record() as led:
        tools.call("x" * 250, {}, errors="return")
    assert not audit.replay(led).drifted


# 0.8.0 machine testing, finding 17 (fixed in 0.9.0): false drift for a tuple or bytearray override.
def test_replay_of_a_scenario_table_with_a_tuple_override_matches():
    with audit.record() as led:
        tools.call("scenario_table", {"tool": "black_scholes", "arguments": AUDIT_BS, "scenarios": {"t": {"K": (1.0, 2.0)}}})
    assert not audit.replay(led).drifted


# 0.8.0 machine testing, finding 18 (fixed in 0.9.0): verify reported a foreign line's own seq.
def test_verify_names_the_position_of_a_foreign_line(tmp_path):
    path = str(tmp_path / "f.jsonl")
    with audit.record(path):
        for ead in (1.0, 2.0, 3.0):
            tools.call("expected_loss", dict(EL, ead=ead))
    lines = open(path).read().splitlines()
    lines.insert(1, '{"seq": 999}')
    open(path, "w").write("\n".join(lines) + "\n")
    assert audit.verify_chain(path).broken_at == 2


# 0.8.0 machine testing, finding 19 (fixed in 0.9.0): an unwritten file ledger raised RP505.
def test_a_file_ledger_that_recorded_nothing_verifies_as_empty(tmp_path):
    with audit.record(str(tmp_path / "none.jsonl")) as led:
        tools.call("riskpy_status", {})
    chain = audit.verify_chain(led)
    assert chain.ok and chain.entries == 0 and chain.head == led.head


# 0.8.0 machine testing, finding 20: resolved in 0.9.0 by documenting the limit (docs/audit.md, "Number spelling"),
# not by changing the digest format, which would make every new ledger fail verify under 0.8 and under any other
# verifier written to the documented format. Formatting-only re-serialisation verifies; re-spelling numbers does not.
def test_a_ledger_reserialised_by_a_javascript_tool_is_the_documented_limit(tmp_path):
    path = str(tmp_path / "run.jsonl")
    with audit.record(path):
        tools.call("expected_loss", EL)

    def js(v):
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, dict):
            return {k: js(x) for k, x in v.items()}
        return [js(x) for x in v] if isinstance(v, list) else v

    out, spaced = str(tmp_path / "js.jsonl"), str(tmp_path / "spaced.jsonl")
    with open(out, "w") as fh, open(spaced, "w") as fh2:
        for line in open(path):
            entry = json.loads(line)
            fh.write(json.dumps(js(entry), separators=(",", ":")) + "\n")
            fh2.write(json.dumps(dict(reversed(list(entry.items()))), indent=None, separators=(" , ", " : ")) + "\n")
    assert audit.verify_chain(spaced).ok                     # key order and spacing are not part of the digest
    chain = audit.verify_chain(out)                          # number spelling is: 9.0 written as 9
    assert not chain.ok and chain.broken_at == 1 and "digest does not match" in chain.reason


# ==============================================================================================================
# Agent tool surface properties (workstream 1)
# (from test_tools_props.py)
# ==============================================================================================================

"""Property tests for the agent tool surface (riskpy.tools), generated from each tool's own JSON schema.

Deterministic under the fixed Hypothesis profile below (derandomize=True, no database), and fast: every
property runs a few dozen examples per tool. The long coverage-guided campaigns that found the strict
xfails at the bottom ran the same generators for tens of thousands of examples (see REPORT.md).
"""

import json
import math
import os
import tempfile

import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

import riskpy.errors as rerrors
import riskpy.tools as tools

settings.register_profile(
    "riskpy-tools", derandomize=True, database=None, deadline=None, max_examples=25, print_blob=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much],
)
TOOLS_PROFILE = settings.get_profile("riskpy-tools")

MANIFEST = {m["name"]: m for m in tools.manifest()}
FUZZED = sorted(n for n in MANIFEST if n != "verify")          # verify alone takes seconds per call
DRIVABLE = sorted(n for n, m in MANIFEST.items()
                  if m["kind"] == "calculator" and n not in ("sensitivity", "scenario_table", "verify"))

# Plausible ranges by parameter name, so a fair share of generated calls succeed and reach deep code.
REAL = {
    "S": (1e-3, 1e5), "K": (1e-3, 1e5), "T": (0.0, 30.0), "r": (-0.05, 0.2), "sigma": (0.0, 1.5), "q": (0.0, 0.1),
    "price": (0.0, 200.0), "face": (1.0, 1e6), "coupon": (0.0, 0.15), "maturity": (0.0, 40.0), "ytm": (-0.02, 0.2),
    "pd": (0.0, 1.0), "lgd": (0.0, 1.0), "ead": (0.0, 1e8), "assets": (1.0, 1e6), "debt": (1.0, 1e6),
    "hazard": (0.0, 0.5), "recovery": (0.0, 1.0), "i": (-0.02, 0.2), "expected_frequency": (0.0, 20.0),
    "severity_mean": (1.0, 1e6), "severity_sd": (1.0, 1e6), "tail": (1.0, 1.3), "kappa": (0.01, 3.0),
    "theta": (0.0, 0.15), "level": (0.5, 0.9999), "rho": (-0.99, 0.99), "expected_loss_ratio": (0.3, 1.5),
    "copula_theta": (0.05, 20.0), "df": (1.0, 60.0), "horizon": (0.1, 30.0), "spread": (0.0, 0.1),
    "rate": (-0.05, 0.3), "bump": (1e-6, 1.0),
}
INTS = {"frequency": (1, 12), "n": (0, 120), "x": (0, 130), "t": (0, 60), "trials": (1, 2000), "seed": (0, 1000),
        "recent": (0, 50)}
CHOICES = {
    "kind": ["call", "put", "whole_life", "term", "endowment"], "table": ["sult"], "model": ["vasicek", "cir"],
    "asset_class": ["corporate", "sovereign", "bank", "retail"], "measure": ["tvar", "var"],
    "method": ["historical", "parametric", "euler", "covariance", "standalone"], "marginal": ["lognormal", "normal"],
    "copula": ["gaussian", "t", "clayton", "survival_clayton", "gumbel", "survival_gumbel", "frank"],
    "family": ["gaussian", "t", "clayton", "survival_clayton", "gumbel", "survival_gumbel", "frank"],
    "from_compounding": ["continuous", "annual", "semiannual", "monthly", "12"],
    "to_compounding": ["continuous", "annual", "quarterly", "daily", "365"],
    "topic": ["RP105", "black_scholes", "RP999"], "modules": ["core", "special"],
}
EDGE = [0.0, -0.0, 5e-324, 1e-300, 1.0, -1.0, 1e15, 1.7976931348623157e308, 2**31 - 1, 2**31, 10**400]
BAD = [math.nan, math.inf, -math.inf, True, None, "1", [], {}, 10**400]


def _scalar(name, integer):
    plausible = (st.integers(*INTS[name]) if name in INTS else
                 st.floats(*REAL.get(name, (-1e6, 1e6)), allow_nan=False, allow_infinity=False))
    if integer and name not in INTS:
        plausible = st.integers(-1000, 1000)
    return st.one_of(plausible, plausible, plausible, st.sampled_from(EDGE), st.sampled_from(BAD))


def _value(name, schema):
    t = schema.get("type")
    if isinstance(t, list):
        return st.one_of(*[_value(name, {"type": x, "items": schema.get("items", {})}) for x in t])
    if "anyOf" in schema:
        return st.one_of(*[_value(name, s) for s in schema["anyOf"]])
    if t in ("number", "integer"):
        return _scalar(name, t == "integer")
    if t == "boolean":
        return st.one_of(st.booleans(), st.sampled_from(BAD))
    if t == "string":
        return st.one_of(st.sampled_from(CHOICES.get(name, ["x"])), st.text(max_size=12),
                         st.sampled_from(["", "a" * 201, "\x1b[31m", "‮", "²"]), st.sampled_from(BAD))
    if t == "null":
        return st.none()
    if t == "array":
        if name in ("correlation", "matrix"):
            return st.integers(0, 5).flatmap(lambda n: st.lists(
                st.lists(st.one_of(st.floats(-1, 1), st.sampled_from([1.0, math.nan, "x"])), min_size=n, max_size=n),
                min_size=n, max_size=n))
        if name == "triangle":
            return st.lists(st.lists(st.one_of(st.floats(1, 1e6), st.none(), st.sampled_from([0.0, -1.0])),
                                     min_size=1, max_size=5), max_size=5)
        sub = {"par_rates": "rate", "means": "severity_mean", "sds": "severity_sd", "standalone": "ead",
               "premiums": "ead", "returns": "q", "maturities": "maturity", "at": "maturity"}.get(name, name)
        return st.lists(_value(sub, schema.get("items") or {"type": "number"}), max_size=6)
    return st.recursive(st.one_of(st.integers(), st.floats(), st.text(max_size=4)),
                        lambda c: st.one_of(st.lists(c, max_size=3), st.dictionaries(st.text(max_size=3), c, max_size=3)),
                        max_leaves=6)


@st.composite
def arguments_for(draw, name, depth=0):
    schema = MANIFEST[name]["input_schema"]
    required = set(schema.get("required", []))
    args = {}
    for p, ps in schema["properties"].items():
        if name in ("sensitivity", "scenario_table") and p in ("tool", "arguments", "parameter", "scenarios"):
            continue
        if p in required or draw(st.booleans()):
            args[p] = draw(_value(p, ps))
    if name in ("sensitivity", "scenario_table"):
        inner = draw(st.sampled_from(DRIVABLE))
        base = draw(arguments_for(inner, depth + 1)) if depth == 0 else {}
        if "seed" in MANIFEST[inner]["input_schema"]["properties"]:
            base["seed"] = draw(st.integers(1, 99))
        params = sorted(MANIFEST[inner]["input_schema"]["properties"])
        args.update({"tool": inner, "arguments": base})
        if name == "sensitivity":
            args["parameter"] = draw(st.sampled_from(params))
        else:
            labels = draw(st.lists(st.text(max_size=6), min_size=1, max_size=4, unique=True))
            args["scenarios"] = {}
            for label in labels:
                p = draw(st.sampled_from(params))
                args["scenarios"][label] = {p: draw(_value(p, MANIFEST[inner]["input_schema"]["properties"][p]))}
    mutation = draw(st.integers(0, 9))
    if mutation == 8 and args:
        args.pop(draw(st.sampled_from(sorted(args))))       # a missing argument
    elif mutation == 9:
        args[draw(st.sampled_from(["vol", "spot", "zzz", ""]))] = 1.0   # an unknown argument
    return args


def _is_error(out):
    return isinstance(out, dict) and set(out) == {"error"}


def _check_contract(name, args, out):
    if _is_error(out):
        e = out["error"]
        assert {"code", "message", "param", "hint", "docs"} <= set(e), e
        assert e["code"] in rerrors.CATALOGUE, e
        assert e["code"] != "RP900", (name, args, e)
        json.dumps(out, allow_nan=False)
    else:
        json.dumps(out, allow_nan=False)        # finite JSON numbers only


@pytest.mark.parametrize("name", FUZZED)
@TOOLS_PROFILE
@given(data=st.data())
def test_tool_contract_on_schema_generated_arguments(name, data):
    """call(errors="return") never raises; a failure is a catalogued code (never RP900); a success is strict JSON."""
    args = data.draw(arguments_for(name))
    out = tools.call(name, args, errors="return")
    _check_contract(name, args, out)


@TOOLS_PROFILE
@given(name=st.sampled_from(FUZZED), junk=st.one_of(st.none(), st.integers(), st.text(max_size=5),
                                                    st.lists(st.integers(), max_size=2)))
def test_non_object_arguments_are_rp406_or_missing(name, junk):
    out = tools.call(name, junk, errors="return")
    if junk is None:
        _check_contract(name, junk, out)
    else:
        assert _is_error(out) and out["error"]["code"] == "RP406", out


def _deep(n):
    v = 1
    for _ in range(n):
        v = {"a": v}
    return v


TOOLS_BS = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}
L = tools.LIMITS
OVER_LIMIT = {
    "string": ("black_scholes", {**TOOLS_BS, "kind": "c" * (L["string"] + 1)}),
    "tool name": ("x" * (L["string"] + 1), {}),
    "argument name": ("black_scholes", {**TOOLS_BS, "y" * (L["string"] + 1): 1}),
    "list": ("value_at_risk", {"returns": [0.01] * (L["list"] + 1)}),
    "integer": ("annuity_certain", {"i": 0.05, "n": L["integer"] + 1}),
    "trials": ("aggregate_loss", {"expected_frequency": 1, "severity_mean": 1, "severity_sd": 1,
                                  "trials": L["trials"] + 1}),
    "simulated_claims": ("aggregate_loss", {"expected_frequency": L["simulated_claims"] / L["trials"] + 1,
                                            "severity_mean": 1, "severity_sd": 1, "trials": L["trials"]}),
    "periods": ("bond_analytics", {"face": 100, "coupon": 0.05, "maturity": (L["periods"] + 1) / 12, "ytm": 0.05,
                                   "frequency": 12}),
    "triangle": ("chain_ladder", {"triangle": [[1.0]] * (L["triangle"] + 1)}),
    "matrix": ("nearest_correlation", {"matrix": [[float(i == j) for j in range(L["matrix"] + 1)]
                                                  for i in range(L["matrix"] + 1)]}),
    "capital_cells": ("capital_allocation", {"means": [1.0] * 21, "sds": [1.0] * 21, "copula": "clayton",
                                             "copula_theta": 1.0, "trials": L["capital_cells"] // 20}),
    "scenarios": ("scenario_table", {"tool": "black_scholes", "arguments": TOOLS_BS,
                                     "scenarios": {f"s{i}": {"r": 0.01} for i in range(L["scenarios"] + 1)}}),
    "depth": ("scenario_table", {"tool": "black_scholes", "arguments": TOOLS_BS, "scenarios": {"a": _deep(L["depth"])}}),
    "meta budget": ("sensitivity", {"tool": "aggregate_loss", "parameter": "severity_mean",
                                    "arguments": {"expected_frequency": 100, "severity_mean": 1, "severity_sd": 1,
                                                  "trials": 300_000, "seed": 1}}),
}


@pytest.mark.parametrize("limit", sorted(OVER_LIMIT))
def test_every_limit_is_refused_with_rp405(limit):
    name, args = OVER_LIMIT[limit]
    out = tools.call(name, args, errors="return")
    assert _is_error(out) and out["error"]["code"] == "RP405", out


pos = lambda lo, hi: st.floats(lo, hi, allow_nan=False, allow_infinity=False)  # noqa: E731


@settings(TOOLS_PROFILE, max_examples=60)
@given(S_=pos(20, 500), K=pos(20, 500), T=pos(0.05, 5), r=pos(-0.02, 0.1), sigma=pos(0.05, 1.0), q=pos(0, 0.05),
       kind=st.sampled_from(["call", "put"]))
def test_sensitivity_of_black_scholes_matches_option_greeks(S_, K, T, r, sigma, q, kind):
    base = {"S": S_, "K": K, "T": T, "r": r, "sigma": sigma, "q": q, "kind": kind}
    g = tools.call("option_greeks", base)
    for param, key, scale in (("S", "delta", 1.0), ("sigma", "vega", 100.0), ("r", "rho", 100.0),
                              ("T", "theta", -365.0)):
        s = tools.call("sensitivity", {"tool": "black_scholes", "arguments": base, "parameter": param})
        assert math.isclose(s["outputs"]["result"]["derivative"], g[key] * scale, rel_tol=2e-4,
                            abs_tol=1e-6 * max(S_, K)), (param, s["outputs"]["result"], g)
    s = tools.call("sensitivity", {"tool": "black_scholes", "arguments": base, "parameter": "S"})
    assert math.isclose(s["outputs"]["result"]["second_derivative"], g["gamma"], rel_tol=2e-3, abs_tol=1e-6)


@settings(TOOLS_PROFILE, max_examples=60)
@given(S_=pos(20, 500), K=pos(20, 500), T=pos(0.05, 5), r=pos(-0.02, 0.1), sigma=pos(0.05, 1.5),
       kind=st.sampled_from(["call", "put"]))
def test_implied_vol_inverts_black_scholes(S_, K, T, r, sigma, kind):
    args = {"S": S_, "K": K, "T": T, "r": r, "sigma": sigma, "kind": kind}
    price = tools.call("black_scholes", args)
    vega = tools.call("option_greeks", args)["vega"]
    assume(price > 1e-6 * S_ and vega > 1e-6)
    iv = tools.call("implied_vol", {"price": price, "S": S_, "K": K, "T": T, "r": r, "kind": kind})
    assert abs(iv - sigma) < 1e-6 + 1e-9 / vega


COMPOUNDING = ["continuous", "annual", "semiannual", "quarterly", "monthly", "weekly", "daily", "12", "365"]


@settings(TOOLS_PROFILE, max_examples=60)
@given(rate=pos(-0.5, 0.9), a=st.sampled_from(COMPOUNDING), b=st.sampled_from(COMPOUNDING))
def test_convert_rate_round_trips(rate, a, b):
    x = tools.call("convert_rate", {"rate": rate, "from_compounding": a, "to_compounding": b}, errors="return")
    assume(not _is_error(x))
    y = tools.call("convert_rate", {"rate": x, "from_compounding": b, "to_compounding": a}, errors="return")
    assume(not _is_error(y))
    assert math.isclose(y, rate, rel_tol=1e-9, abs_tol=1e-12)


@settings(TOOLS_PROFILE, max_examples=15)
@given(freq=pos(0, 20), mean=pos(1, 1e4), sd=pos(1, 1e4), trials=st.integers(2, 2000), seed=st.integers(1, 10**6))
def test_seeded_simulation_is_reproducible_and_identity_scenario_changes_nothing(freq, mean, sd, trials, seed):
    args = {"expected_frequency": freq, "severity_mean": mean, "severity_sd": sd, "trials": trials, "seed": seed}
    a = tools.call("aggregate_loss", args, errors="return")
    assert json.dumps(a) == json.dumps(tools.call("aggregate_loss", args, errors="return"))
    if not _is_error(a):
        tab = tools.call("scenario_table", {"tool": "aggregate_loss", "arguments": args,
                                            "scenarios": {"same": {"seed": seed}}})
        assert all(v == 0.0 for v in tab["scenarios"]["same"]["change"].values())


@settings(TOOLS_PROFILE, max_examples=40)
@given(name=st.sampled_from(["black_scholes", "annuity_certain", "expected_loss", "nope"]), bad=st.booleans())
def test_trace_records_every_call_with_its_outcome(name, bad):
    args = {"black_scholes": TOOLS_BS, "annuity_certain": {"i": 0.05, "n": 10},
            "expected_loss": {"pd": 0.01, "lgd": 0.4, "ead": 100.0}, "nope": {}}[name]
    if bad:
        args = {**args, "zzz": 1}
    before = tools.trace(limit=1)
    out = tools.call(name, args, errors="return")
    last = tools.trace(limit=1)[-1]
    assert last["tool"] == name and (not before or last["seq"] == before[-1]["seq"] + 1)
    assert last["ok"] == (not _is_error(out))
    if _is_error(out):
        assert last["code"] == out["error"]["code"]


# ---------------------------------------------------------------------------
# Confirmed defects (strict xfails: they start passing when the bug is fixed)


# 0.8.0 machine testing, finding 1 (fixed in 0.9.0): null instead of RP304 for an overflowed number.
@pytest.mark.parametrize("name,args", [
    ("option_greeks", {"S": 100.0, "K": 1e300, "T": 1000.0, "r": -0.05, "sigma": 0.2, "kind": "put"}),
    ("option_greeks", {"S": 1e-300, "K": 1e-300, "T": 1.0, "r": 0.0, "sigma": 1e-300}),
    ("basel_irb_capital", {"pd": 0.5, "lgd": 0.75, "ead": 1.7976931348623157e308, "asset_class": "sovereign"}),
])
def test_no_null_in_place_of_an_overflowed_number(name, args):
    out = tools.call(name, args, errors="return")
    assert _is_error(out) or all(v is not None for v in out.values()), out


# 0.8.0 machine testing, finding 6 (fixed in 0.9.0): Mack's zero reserve was a false RP304.
def test_mack_on_a_fully_developed_triangle_answers():
    tri = [[100.0, 100.0, 100.0], [100.0, 100.0, None], [100.0, None, None]]
    out = tools.call("mack_chain_ladder", {"triangle": tri}, errors="return")
    assert not _is_error(out) and out["total_reserve"] == 0.0, out


# 0.8.0 machine testing, finding 2 (fixed in 0.9.0): meta-tools ran seed-0 simulations.
def test_meta_tools_never_run_a_simulation_with_a_fresh_seed():
    from riskpy import audit
    base = {"expected_frequency": 2.0, "severity_mean": 100.0, "severity_sd": 50.0, "trials": 500, "seed": 5}
    path = os.path.join(tempfile.mkdtemp(), "ledger.jsonl")
    with audit.record(path):
        a = tools.call("scenario_table", {"tool": "aggregate_loss", "arguments": base,
                                          "scenarios": {"fresh": {"seed": 0}}}, errors="return")
        b = tools.call("sensitivity", {"tool": "aggregate_loss", "arguments": {**base, "seed": 1},
                                       "parameter": "seed"}, errors="return")
    assert _is_error(a) or "error" in a["scenarios"]["fresh"]
    assert _is_error(b)
    assert not audit.replay(path).drifted


# 0.8.0 machine testing, finding 10 (fixed in 0.9.0): bare errors without a parameter.
@pytest.mark.parametrize("name,args", [
    ("black_scholes", {"S": 100.0, "K": 90.0, "T": 1e-4, "r": 0.05, "sigma": 5e-324}),
    ("aggregate_loss", {"expected_frequency": 1.0, "severity_mean": 1.0, "severity_sd": 5e-324, "trials": 10,
                        "seed": 1}),
    ("sensitivity", {"tool": "vasicek_loss_quantile", "arguments": {"level": 0.99, "pd": 0.01, "rho": 10 ** 400},
                     "parameter": "rho"}),
    ("short_rate_zero_bond", {"model": "cir", "r": 0.05, "T": 1.0, "kappa": 1e-170, "theta": 0.04, "sigma": 1e-170}),
])
def test_failures_name_the_parameter(name, args):
    out = tools.call(name, args, errors="return")
    assert not _is_error(out) or out["error"]["param"] is not None, out


# 0.8.0 machine testing, finding 14 (fixed in 0.9.0): the trace reader raised RecursionError.
def test_trace_file_reader_never_raises(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("[" * 100_000 + "]" * 100_000 + "\n" + '{"seq": 1, "tool": "black_scholes", "ok": true}\n')
    assert len(tools._read_trace_file(str(path), None)) == 1


# 0.8.0 machine testing, finding 8 (fixed in 0.9.0): FrankCopula.cdf collapsed at tiny theta.
@pytest.mark.parametrize("theta", [1e-200, -1e-200])
def test_frank_with_a_tiny_theta_is_the_independence_copula(theta):
    from riskpy import capital
    assert math.isclose(capital.FrankCopula(theta).cdf(0.3, 0.9), 0.27, rel_tol=1e-9)


# 0.8.0 machine testing, finding 11 (fixed in 0.9.0): explain=true refused a price explain=false gives.
def test_explain_never_turns_a_price_into_an_error():
    args = {"S": 1.0, "K": 2.0, "T": 2.220446049250313e-16, "r": 0.0, "sigma": 6.8416506183402196e-304}
    assert tools.call("black_scholes", args) == 0.0
    assert not _is_error(tools.call("black_scholes", {**args, "explain": True}, errors="return"))
