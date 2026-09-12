"""Tests for the interest-rate layer: curves, bonds and short-rate models.

Everything here is pinned to an identity — par bonds price at par, durations
are what calculus says, bootstrapped curves reproduce their inputs, closed-form
bond prices agree with Monte Carlo — rather than to numbers copied from
elsewhere.
"""

from __future__ import annotations

import math

import pytest

from riskpy import rates


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------


def test_discount_factor_conventions():
    assert rates.discount_factor(0.05, 2.0) == pytest.approx(math.exp(-0.10))
    assert rates.discount_factor(0.05, 2.0, "annual") == pytest.approx(1.05 ** -2)
    assert rates.discount_factor(0.05, 2.0, 2) == pytest.approx(1.025 ** -4)
    assert rates.discount_factor(0.05, 0.0) == 1.0


def test_zero_rate_inverts_discount_factor():
    for compounding in ("continuous", "annual", 2, 12):
        df = rates.discount_factor(0.037, 3.5, compounding)
        assert rates.zero_rate(df, 3.5, compounding) == pytest.approx(0.037, abs=1e-14)


def test_convert_rate_round_trips():
    for a in ("continuous", "annual", 2, 4, 12):
        for b in ("continuous", "annual", 2, 4, 12):
            back = rates.convert_rate(rates.convert_rate(0.045, a, b), b, a)
            assert back == pytest.approx(0.045, abs=1e-14)
    assert rates.convert_rate(0.05, "annual", "continuous") == pytest.approx(math.log(1.05))


def test_forward_rate_from_discount_factors():
    df1, df2 = math.exp(-0.03 * 2.0), math.exp(-0.035 * 5.0)
    f = rates.forward_rate(df1, 2.0, df2, 5.0)
    assert df2 == pytest.approx(df1 * math.exp(-f * 3.0))


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def test_flat_curve():
    curve = rates.YieldCurve.flat(0.03)
    assert curve.df(5.0) == pytest.approx(math.exp(-0.15))
    assert curve.zero(0.5) == pytest.approx(0.03)
    assert curve.forward(1.0, 4.0) == pytest.approx(0.03, abs=1e-12)
    assert curve.df(0.0) == 1.0


def test_forward_consistency_on_a_sloped_curve():
    curve = rates.YieldCurve([1, 2, 5, 10, 30], [0.02, 0.025, 0.03, 0.033, 0.035])
    for t1, t2 in ((1.0, 2.0), (2.0, 5.0), (3.3, 7.7), (10.0, 30.0)):
        f = curve.forward(t1, t2)
        assert curve.df(t2) == pytest.approx(curve.df(t1) * math.exp(-f * (t2 - t1)), rel=1e-12)


def test_log_linear_interpolation_has_piecewise_constant_forwards():
    curve = rates.YieldCurve([1, 2, 5], [0.02, 0.03, 0.04], interpolation="log_linear")
    assert curve.forward(2.2, 2.5) == pytest.approx(curve.forward(3.0, 4.5), rel=1e-10)
    assert curve.df(2.0) == pytest.approx(math.exp(-0.06))


def test_extrapolation_is_flat_and_documented():
    curve = rates.YieldCurve([1, 5], [0.02, 0.04])
    assert curve.zero(0.1) == pytest.approx(0.02)
    assert curve.zero(40.0) == pytest.approx(0.04)


def test_nelson_siegel_limits_hold_exactly():
    curve = rates.YieldCurve.nelson_siegel(0.04, -0.02, 0.03, 1.5)
    assert curve.zero(1e-9) == pytest.approx(0.02, abs=1e-8)
    assert curve.zero(1e6) == pytest.approx(0.04, abs=1e-6)
    assert curve.zero(3.0) == pytest.approx(rates.nelson_siegel_zero(3.0, 0.04, -0.02, 0.03, 1.5))
    assert curve.df(3.0) == pytest.approx(math.exp(-3.0 * curve.zero(3.0)))
    sv = rates.YieldCurve.svensson(0.04, -0.02, 0.03, 0.01, 1.5, 8.0)
    assert sv.zero(1e-9) == pytest.approx(0.02, abs=1e-8)
    assert sv.zero(1e6) == pytest.approx(0.04, abs=1e-6)


def test_bootstrap_reproduces_par_rates():
    par = [0.020, 0.025, 0.028, 0.030, 0.031, 0.0315]
    maturities = [1, 2, 3, 4, 5, 6]
    for frequency in (1, 2):
        curve = rates.YieldCurve.bootstrap(par, maturities, frequency=frequency)
        for m, p in zip(maturities, par):
            assert curve.par_rate(m, frequency) == pytest.approx(p, abs=1e-10)


def test_from_discount_factors_round_trip():
    times = [0.5, 1.0, 3.0]
    dfs = [0.99, 0.975, 0.91]
    curve = rates.YieldCurve.from_discount_factors(times, dfs)
    for t, df in zip(times, dfs):
        assert curve.df(t) == pytest.approx(df, rel=1e-12)


def test_shift_and_points():
    curve = rates.YieldCurve([1, 5], [0.02, 0.04])
    up = curve.shift(25)
    assert up.zero(5.0) == pytest.approx(0.0425)
    times, zeros, forwards = curve.points()
    assert len(times) == len(zeros) == len(forwards) > 2


def test_curve_validation():
    with pytest.raises(ValueError):
        rates.YieldCurve([], [])
    with pytest.raises(ValueError):
        rates.YieldCurve([1, 2], [0.02])
    with pytest.raises(ValueError):
        rates.YieldCurve([2, 1], [0.02, 0.03])
    with pytest.raises(ValueError):
        rates.YieldCurve([1, 2], [0.02, 0.03], interpolation="spline")
    with pytest.raises(ValueError):
        rates.YieldCurve([0.0, 1.0], [0.02, 0.03])


# ---------------------------------------------------------------------------
# Bonds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frequency", [1, 2, 4])
def test_par_bond_prices_at_par(frequency):
    bond = rates.Bond(face=100.0, coupon=0.05, maturity=10.0, frequency=frequency)
    assert bond.price(0.05) == pytest.approx(100.0, abs=1e-9)


def test_yield_round_trip_and_price_monotonicity():
    bond = rates.Bond(face=100.0, coupon=0.04, maturity=7.0, frequency=2)
    for y in (0.001, 0.02, 0.037, 0.08, 0.15):
        assert bond.yield_to_maturity(bond.price(y)) == pytest.approx(y, abs=1e-10)
    prices = [bond.price(y) for y in (0.01, 0.03, 0.05, 0.07)]
    assert prices == sorted(prices, reverse=True)


def test_zero_coupon_durations():
    zero = rates.Bond(face=100.0, coupon=0.0, maturity=7.0, frequency=1)
    assert zero.macaulay_duration(0.04) == pytest.approx(7.0, abs=1e-12)
    assert zero.modified_duration(0.04) == pytest.approx(7.0 / 1.04, abs=1e-12)
    semi = rates.Bond(face=100.0, coupon=0.06, maturity=5.0, frequency=2)
    assert semi.modified_duration(0.05) == pytest.approx(semi.macaulay_duration(0.05) / 1.025, abs=1e-12)


def test_dv01_and_convexity_against_bumped_prices():
    bond = rates.Bond(face=100.0, coupon=0.06, maturity=8.0, frequency=2)
    y, h = 0.045, 1e-4
    numerical = -(bond.price(y + h) - bond.price(y - h)) / (2 * h) * 1e-4
    assert bond.dv01(y) == pytest.approx(numerical, rel=1e-6)
    assert bond.dv01(y) > 0.0
    second = (bond.price(y + h) - 2 * bond.price(y) + bond.price(y - h)) / (h * h)
    assert bond.convexity(y) == pytest.approx(second / bond.price(y), rel=1e-4)


def test_second_order_approximation_beats_first_order():
    bond = rates.Bond(face=100.0, coupon=0.05, maturity=10.0, frequency=2)
    y, dy = 0.05, 0.01
    price = bond.price(y)
    actual = bond.price(y + dy) - price
    first = -bond.modified_duration(y) * price * dy
    second = rates.duration_price_change(price, bond.modified_duration(y), bond.convexity(y), dy)
    assert abs(second - actual) < abs(first - actual)
    assert second == pytest.approx(actual, rel=0.01)


def test_price_from_curve_and_z_spread():
    curve = rates.YieldCurve.flat(0.03)
    bond = rates.Bond(face=100.0, coupon=0.04, maturity=5.0, frequency=2)
    fair = bond.price_from_curve(curve)
    assert bond.z_spread(fair, curve) == pytest.approx(0.0, abs=1e-10)
    shifted = bond.price_from_curve(curve.shift(150))
    assert bond.z_spread(shifted, curve) == pytest.approx(0.015, abs=1e-8)


def test_bond_price_function_and_cashflows():
    assert rates.bond_price(100.0, 0.05, 10.0, 0.05, 2) == pytest.approx(100.0, abs=1e-9)
    flows = rates.Bond(100.0, 0.04, 2.0, 2).cashflows()
    assert len(flows) == 4
    assert flows[-1] == (2.0, pytest.approx(102.0))
    assert sum(amount for _, amount in flows) == pytest.approx(108.0)


def test_bond_validation():
    with pytest.raises(ValueError):
        rates.Bond(face=-1.0, coupon=0.05, maturity=5.0)
    with pytest.raises(ValueError):
        rates.Bond(face=100.0, coupon=0.05, maturity=0.0)
    with pytest.raises(ValueError):
        rates.Bond(face=100.0, coupon=0.05, maturity=5.0, frequency=0)


# ---------------------------------------------------------------------------
# Short-rate models
# ---------------------------------------------------------------------------


def _deterministic_integral(r0, kappa, theta, T):
    """∫₀ᵀ r(t) dt for the mean-reverting ODE dr = κ(θ − r)dt."""
    return theta * T + (r0 - theta) * (1.0 - math.exp(-kappa * T)) / kappa


def test_zero_bonds_at_zero_volatility_are_deterministic():
    r0, kappa, theta, T = 0.03, 0.5, 0.05, 6.0
    expected = math.exp(-_deterministic_integral(r0, kappa, theta, T))
    assert rates.vasicek_zero_bond(r0, 0.0, T, kappa, theta, 0.0) == pytest.approx(expected, rel=1e-12)
    assert rates.cir_zero_bond(r0, 0.0, T, kappa, theta, 0.0) == pytest.approx(expected, rel=1e-12)
    # The closed form has a 1/sigma² exponent, which the naive formula turns
    # into garbage below about sigma = 1e-6; the log1p form does not.
    assert rates.cir_zero_bond(r0, 0.0, T, kappa, theta, 1e-9) == pytest.approx(expected, rel=1e-9)
    # The sigma^2 correction is second order, so a small sigma is close.
    assert rates.cir_zero_bond(r0, 0.0, T, kappa, theta, 0.01) == pytest.approx(expected, rel=1e-3)
    assert rates.vasicek_zero_bond(r0, 0.0, T, kappa, theta, 0.01) == pytest.approx(expected, rel=1e-3)


def test_vasicek_paths_have_the_exact_transition_moments():
    np = pytest.importorskip("numpy")
    r0, kappa, theta, sigma, T = 0.02, 0.8, 0.05, 0.015, 3.0
    paths = rates.vasicek_paths(r0, kappa, theta, sigma, T, steps=60, trials=60_000, seed=9)
    assert paths.shape == (60_000, 61)
    assert np.all(paths[:, 0] == r0)
    mean = theta + (r0 - theta) * math.exp(-kappa * T)
    var = sigma * sigma / (2 * kappa) * (1.0 - math.exp(-2 * kappa * T))
    terminal = paths[:, -1]
    assert terminal.mean() == pytest.approx(mean, abs=4.0 * math.sqrt(var / 60_000))
    assert terminal.var() == pytest.approx(var, rel=0.03)


def test_vasicek_bond_agrees_with_monte_carlo():
    np = pytest.importorskip("numpy")
    r0, kappa, theta, sigma, T = 0.03, 0.5, 0.04, 0.02, 5.0
    exact = rates.vasicek_zero_bond(r0, 0.0, T, kappa, theta, sigma)
    paths = rates.vasicek_paths(r0, kappa, theta, sigma, T, steps=500, trials=40_000, seed=7)
    dt = T / 500
    # Trapezoid on the integral; the residual discretisation bias is O(dt),
    # which at dt = 0.01 is a few parts in ten thousand, hence the tolerance.
    integral = (0.5 * (paths[:, 0] + paths[:, -1]) + paths[:, 1:-1].sum(axis=1)) * dt
    mc = float(np.exp(-integral).mean())
    assert mc == pytest.approx(exact, rel=2e-3)


def test_cir_paths_stay_non_negative_and_warn_on_feller():
    np = pytest.importorskip("numpy")
    paths = rates.cir_paths(0.03, 0.5, 0.04, 0.1, 2.0, steps=100, trials=5_000, seed=3)
    assert paths.min() >= 0.0
    with pytest.warns(RuntimeWarning, match="Feller"):
        rates.cir_paths(0.03, 0.2, 0.02, 0.3, 1.0, steps=10, trials=100, seed=1)


def test_cir_bond_agrees_with_monte_carlo():
    np = pytest.importorskip("numpy")
    r0, kappa, theta, sigma, T = 0.03, 0.6, 0.04, 0.05, 4.0
    exact = rates.cir_zero_bond(r0, 0.0, T, kappa, theta, sigma)
    paths = rates.cir_paths(r0, kappa, theta, sigma, T, steps=400, trials=40_000, seed=11)
    dt = T / 400
    integral = (0.5 * (paths[:, 0] + paths[:, -1]) + paths[:, 1:-1].sum(axis=1)) * dt
    assert float(np.exp(-integral).mean()) == pytest.approx(exact, rel=3e-3)


def test_model_curves_match_their_bond_prices():
    r0, kappa, theta, sigma = 0.03, 0.5, 0.04, 0.02
    curve = rates.vasicek_curve(r0, kappa, theta, sigma)
    for T in (0.5, 3.0, 10.0):
        assert curve.df(T) == pytest.approx(rates.vasicek_zero_bond(r0, 0.0, T, kappa, theta, sigma), rel=1e-12)
    cir = rates.cir_curve(r0, kappa, theta, sigma)
    for T in (0.5, 3.0, 10.0):
        assert cir.df(T) == pytest.approx(rates.cir_zero_bond(r0, 0.0, T, kappa, theta, sigma), rel=1e-12)


def test_short_rate_validation():
    with pytest.raises(ValueError):
        rates.vasicek_zero_bond(0.03, 2.0, 1.0, 0.5, 0.04, 0.02)  # t > T
    with pytest.raises(ValueError):
        rates.vasicek_paths(0.03, 0.5, 0.04, -0.02, 1.0)
    with pytest.raises(ValueError):
        rates.cir_paths(0.03, 0.5, 0.04, 0.02, 1.0, steps=0)


def test_verification_checks_all_pass():
    for name, value, reference, tolerance in rates._verification_checks():
        assert abs(value - reference) <= tolerance, name
