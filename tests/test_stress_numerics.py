"""Stress and accuracy tests for the special functions, option pricing and rates.

Each regression test pins a bug found by edge-case probing of 0.5.2; the
randomised tests sweep wide parameter ranges against SciPy or a closed form,
with fixed seeds so a failure reproduces.
"""

import math
import random

import pytest

import riskpy
from riskpy import _special as sp
from riskpy import quant, rates
from riskpy.errors import ConvergenceError, InputError


def _rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a)


# ---------------------------------------------------------------------------
# Incomplete gamma / beta at large parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a, x", [(1e5, 1e5), (1e6, 1e6), (1e6, 0.999e6), (1e6, 1.001e6), (1e7, 1e7)])
def test_gammainc_large_shape_is_not_truncated(a, x):
    special = pytest.importorskip("scipy.special")
    # A fixed 1000-iteration cap gave P(1e6, 1e6) = 0.34 instead of 0.5.
    assert sp.gammainc(a, x) == pytest.approx(float(special.gammainc(a, x)), rel=1e-9)
    assert sp.gammaincc(a, x) == pytest.approx(float(special.gammaincc(a, x)), rel=1e-9)
    assert sp.gammainc(a, x) + sp.gammaincc(a, x) == pytest.approx(1.0, abs=1e-14)


def test_gammainc_vec_matches_scalar_at_large_shape():
    np = pytest.importorskip("numpy")
    xs = np.linspace(0.995e6, 1.005e6, 21)
    vec = sp.gammainc_vec(1e6, xs)
    assert vec == pytest.approx([sp.gammainc(1e6, float(x)) for x in xs], rel=1e-12, abs=1e-300)


def test_chi2_large_df_and_tiny_quantiles():
    stats = pytest.importorskip("scipy.stats")
    assert sp.chi2_cdf(1e6, 1e6) == pytest.approx(float(stats.chi2.cdf(1e6, 1e6)), rel=1e-9)
    assert sp.chi2_ppf(0.5, 1e6) == pytest.approx(float(stats.chi2.ppf(0.5, 1e6)), rel=1e-12)
    # An absolute 1e-12 tolerance returned 5.7e-13 for a quantile of 7e-61.
    assert sp.chi2_ppf(0.5, 0.01) == pytest.approx(float(stats.chi2.ppf(0.5, 0.01)), rel=1e-9)
    p = 1 - 1e-12  # not exactly representable: the true complement is 1.0000889e-12
    assert sp.chi2_ppf(p, 10) == pytest.approx(float(stats.chi2.isf(1.0 - p, 10)), rel=1e-9)


def test_gammainc_random_sweep_against_scipy():
    special = pytest.importorskip("scipy.special")
    rng = random.Random(11)
    for _ in range(400):
        a = 10 ** rng.uniform(-1, 6)
        x = a * 10 ** rng.uniform(-0.3, 0.3)
        p, q = sp.gammainc(a, x), sp.gammaincc(a, x)
        assert p == pytest.approx(float(special.gammainc(a, x)), rel=1e-8, abs=1e-14), (a, x)
        assert q == pytest.approx(float(special.gammaincc(a, x)), rel=1e-8, abs=1e-14), (a, x)


# ---------------------------------------------------------------------------
# Student-t
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p, df", [(1e-20, 5), (1e-12, 5), (1e-100, 3), (0.975, 30), (1e-5, 0.5), (0.3, 2.0)])
def test_t_ppf_keeps_the_lower_tail(p, df):
    stats = pytest.importorskip("scipy.stats")
    # Solving t_cdf = 1 - p rounded 1 - 1e-20 to 1 and returned -4096 for -15684.
    assert sp.t_ppf(p, df) == pytest.approx(float(stats.t.ppf(p, df)), rel=1e-9)


def test_t_ppf_just_off_the_centre():
    # SciPy itself returns 0.0 here. Near 0, t ≈ (p - ½)/f(0) with
    # f(0) = Γ((df+1)/2) / (√(df·π)·Γ(df/2)).
    for df in (1.0, 4.0, 50.0):
        f0 = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
        for p in (0.5 + 1e-12, 0.5 - 3e-10, 0.5 + 2e-15):
            assert sp.t_ppf(p, df) == pytest.approx((p - 0.5) / f0, rel=1e-6)


def test_t_ppf_cauchy_far_tail_and_symmetry():
    # df = 1 is the Cauchy distribution: the quantile is tan(π(p - ½)) = -1/(π p) for tiny p.
    assert sp.t_ppf(1e-300, 1) == pytest.approx(-1.0 / (math.pi * 1e-300), rel=1e-9)
    for p, df in [(1e-8, 2.5), (0.1, 7), (0.3, 1.5), (0.4999, 30)]:
        upper = 1 - p
        assert sp.t_ppf(upper, df) == pytest.approx(-sp.t_ppf(1.0 - upper, df), rel=1e-12)


def test_t_ppf_beyond_double_range_is_infinite():
    # p = 1e-250 at df = 0.3 has a quantile near 1e833.
    assert sp.t_ppf(1e-250, 0.3) == -math.inf
    assert sp.t_ppf(1 - 1e-16, 0.05) == math.inf


def test_t_cdf_huge_t_does_not_overflow():
    # |t| > 1e154 overflowed t² and gave a tail of exactly 0.
    tail = sp.t_cdf(-1e300, 1)
    assert tail == pytest.approx(1.0 / (math.pi * 1e300), rel=1e-9)
    assert sp.t_cdf(1e300, 1) == 1.0
    assert sp.t_cdf(-math.inf, 3) == 0.0 and sp.t_cdf(math.inf, 3) == 1.0


def test_t_cdf_random_sweep_including_huge_df():
    stats = pytest.importorskip("scipy.stats")
    rng = random.Random(5)
    for _ in range(600):
        df = 10 ** rng.uniform(-0.5, 12)
        x = rng.choice([-1, 1]) * 10 ** rng.uniform(-3, 1.5)
        got = sp.t_cdf(x, df)
        ref = float(stats.t.cdf(x, df))
        assert got == pytest.approx(ref, rel=1e-9, abs=1e-300), (x, df)


def test_t_ppf_round_trip_random():
    rng = random.Random(9)
    for _ in range(300):
        df = 10 ** rng.uniform(-0.3, 8)
        p = 10 ** rng.uniform(-150, -0.31)
        t = sp.t_ppf(p, df)
        if math.isinf(t):
            continue
        assert sp.t_cdf(t, df) == pytest.approx(p, rel=1e-9), (p, df)


# ---------------------------------------------------------------------------
# Normal, beta, trigamma edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [5e-324, 1e-320, 1e-316, 1e-310])
def test_norm_ppf_subnormal(p):
    stats = pytest.importorskip("scipy.stats")
    # The Halley step's exp(x²/2) overflowed below p ≈ 1e-316.
    assert sp.norm_ppf(p) == pytest.approx(float(stats.norm.ppf(p)), rel=1e-12)


def test_beta_function_sign_for_negative_arguments():
    special = pytest.importorskip("scipy.special")
    for a, b in [(-0.5, 1.0), (-1.5, 0.7), (-2.5, 4.0), (2.0, 3.0), (0.3, 0.4)]:
        assert sp.beta_function(a, b) == pytest.approx(float(special.beta(a, b)), rel=1e-12)


def test_trigamma_tiny_argument():
    assert sp.trigamma(1e-100) == pytest.approx(1e200, rel=1e-12)
    assert sp.trigamma(1e-300) == math.inf


# ---------------------------------------------------------------------------
# Black–Scholes, Greeks, implied volatility
# ---------------------------------------------------------------------------


def test_black_scholes_far_tails_keep_relative_precision():
    stats = pytest.importorskip("scipy.stats")
    assert quant.norm_cdf(-10) == pytest.approx(float(stats.norm.cdf(-10)), rel=1e-13)
    # A put struck at 30% of spot, 6 months, 20% vol: about 7.5e-18, not 0.
    d1 = (math.log(100 / 30) + 0.5 * 0.04 * 0.5) / (0.2 * math.sqrt(0.5))
    ref = 30 * float(stats.norm.cdf(-(d1 - 0.2 * math.sqrt(0.5)))) - 100 * float(stats.norm.cdf(-d1))
    assert quant.black_scholes(100, 30, 0.5, 0, 0.2, kind="put") == pytest.approx(ref, rel=1e-6)
    assert quant.black_scholes(100, 30, 0.5, 0, 0.2, kind="put") > 0


def test_black_scholes_put_call_parity_sweep():
    rng = random.Random(21)
    for _ in range(2000):
        S, K = 100.0, 100 * 10 ** rng.uniform(-0.6, 0.6)
        T, r, q = 10 ** rng.uniform(-3, 1.3), rng.uniform(-0.03, 0.1), rng.uniform(0, 0.05)
        sigma = 10 ** rng.uniform(-3, 0.5)
        c = quant.black_scholes(S, K, T, r, sigma, q, "call")
        p = quant.black_scholes(S, K, T, r, sigma, q, "put")
        parity = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert c - p == pytest.approx(parity, abs=1e-10 * max(S, K))
        assert c >= -1e-12 and p >= -1e-12


@pytest.mark.parametrize("bad", [
    dict(S=math.nan), dict(sigma=math.nan), dict(T=math.nan), dict(r=math.nan), dict(K=math.inf),
])
def test_black_scholes_rejects_non_finite(bad):
    args = dict(S=100, K=100, T=1, r=0.05, sigma=0.2) | bad
    with pytest.raises(InputError) as info:
        quant.black_scholes(**args)
    assert info.value.code == "RP101"


def test_greeks_against_finite_differences():
    rng = random.Random(4)
    h = 1e-4
    for _ in range(200):
        S, K = 100.0, 100 * 10 ** rng.uniform(-0.3, 0.3)
        T, r, q, sigma = rng.uniform(0.05, 3), rng.uniform(-0.01, 0.08), rng.uniform(0, 0.04), rng.uniform(0.05, 0.8)
        kind = rng.choice(["call", "put"])
        g = quant.greeks(S, K, T, r, sigma, q, kind)
        bs = lambda **kw: quant.black_scholes(**({"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q, "kind": kind} | kw))
        assert g.delta == pytest.approx((bs(S=S + h) - bs(S=S - h)) / (2 * h), abs=1e-6)
        assert g.vega == pytest.approx((bs(sigma=sigma + h) - bs(sigma=sigma - h)) / (2 * h) / 100, abs=1e-6)
        assert g.rho == pytest.approx((bs(r=r + h) - bs(r=r - h)) / (2 * h) / 100, abs=1e-6)
        assert g.theta == pytest.approx(-(bs(T=T + h) - bs(T=T - h)) / (2 * h) / 365, abs=1e-6)


def test_greeks_at_zero_vol_follow_the_forward():
    # Forward 100·e^(-0.1) = 90.5 < 95: the call is worthless, so delta is 0.
    assert quant.greeks(100, 95, 1, 0, 0, q=0.10).delta == 0.0
    # Forward 100 > 105·e^(-0.1) = 95: the put is worthless.
    assert quant.greeks(100, 105, 1, 0.10, 0, kind="put").delta == 0.0
    g = quant.greeks(100, 99, 1, 0.05, 0)
    assert g.delta == 1.0
    assert g.theta == pytest.approx(-0.05 * 99 * math.exp(-0.05) / 365, rel=1e-12)
    assert g.rho == pytest.approx(99 * math.exp(-0.05) / 100, rel=1e-12)
    h = 1e-6
    fd_rho = (quant.black_scholes(100, 99, 1, 0.05 + h, 0) - quant.black_scholes(100, 99, 1, 0.05 - h, 0)) / (2 * h) / 100
    assert g.rho == pytest.approx(fd_rho, rel=1e-6)


def test_implied_vol_tiny_price_is_not_converged_at_first_midpoint():
    price = quant.black_scholes(100, 200, 0.25, 0, 0.2)
    assert price < 1e-8
    assert quant.implied_vol(price, 100, 200, 0.25, 0) == pytest.approx(0.2, rel=1e-8)


def test_implied_vol_refuses_a_price_that_does_not_pin_down_vol():
    # At 2000% vol the call is worth S to sixteen digits; bisection returned 15.
    with pytest.raises(ConvergenceError) as info:
        quant.implied_vol(quant.black_scholes(100, 100, 1, 0.05, 20), 100, 100, 1, 0.05)
    assert info.value.code == "RP302"
    with pytest.raises(ConvergenceError) as info:
        quant.implied_vol(0.0, 100, 30, 0.5, 0.03, kind="put")
    assert info.value.code == "RP301"
    with pytest.raises(InputError):
        quant.implied_vol(math.nan, 100, 100, 1, 0.05)


def test_implied_vol_round_trip_within_conditioning():
    """Every solvable quote round-trips to within what double precision allows.

    The attainable vol accuracy is about eps·price/vega: deep in the money the
    time value is a tiny fraction of the price and the quote itself only
    determines the vol that well.
    """
    rng = random.Random(2)
    solved = 0
    for _ in range(1500):
        sigma = 10 ** rng.uniform(-2, 0.7)
        K, T = 100 * 10 ** rng.uniform(-0.7, 0.7), 10 ** rng.uniform(-2, 1)
        kind, r = rng.choice(["call", "put"]), rng.uniform(-0.02, 0.08)
        price = quant.black_scholes(100, K, T, r, sigma, kind=kind)
        try:
            iv = quant.implied_vol(price, 100, K, T, r, kind=kind)
        except ConvergenceError:
            continue
        solved += 1
        vega = quant.greeks(100, K, T, r, sigma, kind=kind).vega * 100
        bound = 64 * 2.2e-16 * price / max(vega, 1e-300) + 1e-9 * sigma
        assert abs(iv - sigma) <= bound, (sigma, K, T, kind, price, iv)
    assert solved > 1000


# ---------------------------------------------------------------------------
# Portfolio risk and paths: validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("call, code", [
    (lambda: quant.expected_shortfall([]), "RP111"),
    (lambda: quant.expected_shortfall([0.01, -0.02], 1.5), "RP110"),
    (lambda: quant.expected_shortfall([0.01, math.nan]), "RP101"),
    (lambda: quant.historical_var([0.01, math.inf]), "RP101"),
    (lambda: quant.parametric_var(mean=0, sd=-0.1), "RP102"),
    (lambda: quant.gbm_paths(100, 0.05, -0.2, 1, trials=10), "RP102"),
    (lambda: quant.gbm_paths(100, 0.05, 0.2, -1, trials=10), "RP106"),
    (lambda: quant.merton_jump_paths(100, 0.05, 0.2, 1, 1, 0, 0.1, steps=0), "RP107"),
    (lambda: quant.merton_jump_paths(100, 0.05, 0.2, 1, 1, 0, -0.1, trials=10), "RP102"),
])
def test_risk_and_path_inputs_are_validated(call, code):
    pytest.importorskip("numpy")
    with pytest.raises(InputError) as info:
        call()
    assert info.value.code == code


def test_expected_shortfall_is_at_least_var():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    for _ in range(50):
        returns = rng.standard_t(3, size=500) * 0.01
        for level in (0.9, 0.95, 0.99):
            assert quant.expected_shortfall(returns, level) >= quant.historical_var(returns, level) - 1e-12


# ---------------------------------------------------------------------------
# Heston
# ---------------------------------------------------------------------------


def test_heston_expiry_is_intrinsic_and_inputs_are_validated():
    pytest.importorskip("numpy")
    assert quant.heston_price(100, 110, 0, 0.03, 0.04, 1.5, 0.04, 0.3, -0.7) == 0.0
    assert quant.heston_price(120, 110, 0, 0.03, 0.04, 1.5, 0.04, 0.3, -0.7) == 10.0
    for bad in (dict(xi=0.0), dict(v0=-0.01), dict(rho=2.0), dict(K=-1.0), dict(kappa=math.nan)):
        args = dict(S=100, K=100, T=1, r=0.03, v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7) | bad
        with pytest.raises(InputError):
            quant.heston_price(**args)


def test_heston_approaches_black_scholes_as_vol_of_vol_vanishes():
    pytest.importorskip("numpy")
    # With v0 = theta, rho = 0 and tiny xi, variance stays at theta: Black–Scholes at
    # sqrt(theta). (With rho != 0 the price moves at first order in xi — that is
    # the skew, not an error.)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        h = quant.heston_price(100, 105, 1, 0.03, 0.04, 2.0, 0.04, 1e-3, 0.0)
    assert h == pytest.approx(quant.black_scholes(100, 105, 1, 0.03, 0.2), abs=1e-5)


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


def test_shifted_parametric_curve_keeps_its_formula():
    curve = rates.YieldCurve.nelson_siegel(0.04, -0.02, 0.01, 2)
    shifted = curve.shift(100)
    for t in (0.0, 0.3, 5.0, 60.0, 200.0):
        assert shifted.zero(t) == pytest.approx(curve.zero(t) + 0.01, abs=1e-15)
    assert shifted.zero(0) == pytest.approx(0.03, abs=1e-15)


def test_vasicek_and_cir_curves_evaluate_their_formula():
    v = rates.vasicek_curve(0.03, 0.5, 0.05, 0.01)
    assert v.zero(0) == 0.03
    assert v.zero(60) == pytest.approx(-math.log(rates.vasicek_zero_bond(0.03, 0, 60, 0.5, 0.05, 0.01)) / 60, rel=1e-14)
    c = rates.cir_curve(0.03, 0.5, 0.05, 0.05)
    assert c.zero(0) == 0.03
    assert c.zero(45) == pytest.approx(-math.log(rates.cir_zero_bond(0.03, 0, 45, 0.5, 0.05, 0.05)) / 45, rel=1e-14)


@pytest.mark.parametrize("kappa", [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 9.9e-4, 1.01e-3, 1e-2, 0.5, 5.0])
def test_vasicek_small_kappa_has_no_cancellation(kappa):
    # The κ → 0 limit is Ho–Lee/Merton: ln P = -rτ - ... ; compare against a
    # high-order series in κτ rather than the cancelling closed form.
    r, theta, sigma = 0.03, 0.05, 0.01
    for tau in (0.1, 5.0, 30.0):
        got = rates.vasicek_zero_bond(r, 0, tau, kappa, theta, sigma)
        assert 0 < got < 1
        if kappa * tau < 1e-6:
            # Leading terms: -rτ + σ²τ³/6 (drift terms are O(κ)).
            assert math.log(got) == pytest.approx(-r * tau + sigma ** 2 * tau ** 3 / 6, rel=1e-5)
    # Continuity across the series/direct switch at κτ = 1e-3 (κ itself moves by 2e-12).
    lo = rates.vasicek_zero_bond(r, 0, 1.0, 1e-3 * (1 - 1e-9), theta, sigma)
    hi = rates.vasicek_zero_bond(r, 0, 1.0, 1e-3 * (1 + 1e-9), theta, sigma)
    assert lo == pytest.approx(hi, rel=1e-13)


def test_discount_factor_overflow_is_a_structured_error():
    with pytest.raises(InputError) as info:
        rates.discount_factor(-10, 1000)
    assert info.value.code == "RP304"
    with pytest.raises(InputError) as info:
        rates.discount_factor(-0.99, 5000, "annual")
    assert info.value.code == "RP304"


def test_discount_factor_round_trips_with_zero_rate():
    rng = random.Random(8)
    for _ in range(500):
        rate, t = rng.uniform(-0.05, 0.3), 10 ** rng.uniform(-2, 2)
        comp = rng.choice(["continuous", "annual", 2, 4, 12, 365])
        df = rates.discount_factor(rate, t, comp)
        assert rates.zero_rate(df, t, comp) == pytest.approx(rate, abs=1e-12)


def test_package_still_imports_cleanly():
    assert riskpy.__version__
