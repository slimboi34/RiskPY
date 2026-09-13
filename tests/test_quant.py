"""Tests for the quantitative finance layer.

The closed-form functions are checked against published values and against
identities that must hold exactly (put-call parity, the Greeks as numerical
derivatives of the price). Identities are the stronger test: a textbook number
catches a typo, but parity catches a whole class of sign errors.
"""

from __future__ import annotations

import math

import pytest

from riskpy import quant


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------


def test_known_black_scholes_value():
    # The canonical worked example: at-the-money, one year, 5% rates, 20% vol.
    price = quant.black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2, kind="call")
    assert price == pytest.approx(10.450583572, abs=1e-6)


def test_known_put_value():
    price = quant.black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2, kind="put")
    assert price == pytest.approx(5.573526022, abs=1e-6)


def test_put_call_parity():
    S, K, T, r, sigma, q = 103.0, 95.0, 0.75, 0.031, 0.27, 0.012
    call = quant.black_scholes(S, K, T, r, sigma, q, "call")
    put = quant.black_scholes(S, K, T, r, sigma, q, "put")
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert call - put == pytest.approx(expected, abs=1e-10)


def test_price_is_monotone_in_volatility():
    prices = [
        quant.black_scholes(S=100, K=100, T=1.0, r=0.02, sigma=v)
        for v in (0.05, 0.1, 0.2, 0.4, 0.8)
    ]
    assert prices == sorted(prices)


def test_zero_volatility_is_discounted_intrinsic():
    price = quant.black_scholes(S=120, K=100, T=2.0, r=0.05, sigma=0.0)
    assert price == pytest.approx(120.0 - 100.0 * math.exp(-0.10), abs=1e-12)


def test_at_expiry_is_intrinsic():
    assert quant.black_scholes(S=120, K=100, T=0.0, r=0.05, sigma=0.3) == pytest.approx(20.0)
    assert quant.black_scholes(S=80, K=100, T=0.0, r=0.05, sigma=0.3) == pytest.approx(0.0)
    assert quant.black_scholes(S=80, K=100, T=0.0, r=0.05, sigma=0.3, kind="put") == pytest.approx(20.0)


def test_deep_out_of_the_money_is_nearly_worthless():
    price = quant.black_scholes(S=100, K=400, T=0.25, r=0.02, sigma=0.15)
    assert 0.0 <= price < 1e-6


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        quant.black_scholes(S=-1, K=100, T=1, r=0.05, sigma=0.2)
    with pytest.raises(ValueError):
        quant.black_scholes(S=100, K=100, T=1, r=0.05, sigma=-0.2)
    with pytest.raises(ValueError):
        quant.black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, kind="straddle")


# ---------------------------------------------------------------------------
# Greeks — checked as numerical derivatives of the price
# ---------------------------------------------------------------------------


S, K, T, r, SIGMA, Q = 100.0, 95.0, 0.6, 0.03, 0.28, 0.01


def test_delta_matches_a_bumped_price():
    h = 1e-5
    up = quant.black_scholes(S + h, K, T, r, SIGMA, Q, "call")
    down = quant.black_scholes(S - h, K, T, r, SIGMA, Q, "call")
    assert quant.greeks(S, K, T, r, SIGMA, Q, "call").delta == pytest.approx(
        (up - down) / (2 * h), rel=1e-5
    )


def test_gamma_matches_the_second_derivative():
    h = 1e-3
    up = quant.black_scholes(S + h, K, T, r, SIGMA, Q, "call")
    mid = quant.black_scholes(S, K, T, r, SIGMA, Q, "call")
    down = quant.black_scholes(S - h, K, T, r, SIGMA, Q, "call")
    numerical = (up - 2 * mid + down) / (h * h)
    assert quant.greeks(S, K, T, r, SIGMA, Q, "call").gamma == pytest.approx(numerical, rel=1e-3)


def test_vega_is_quoted_per_percentage_point():
    h = 1e-6
    up = quant.black_scholes(S, K, T, r, SIGMA + h, Q, "call")
    down = quant.black_scholes(S, K, T, r, SIGMA - h, Q, "call")
    raw = (up - down) / (2 * h)
    assert quant.greeks(S, K, T, r, SIGMA, Q, "call").vega == pytest.approx(raw / 100.0, rel=1e-4)


def test_theta_is_quoted_per_day_and_is_negative_for_a_long_call():
    g = quant.greeks(S, K, T, r, SIGMA, Q, "call")
    h = 1e-6
    later = quant.black_scholes(S, K, T - h, r, SIGMA, Q, "call")
    now = quant.black_scholes(S, K, T, r, SIGMA, Q, "call")
    raw_per_year = (later - now) / h
    assert g.theta == pytest.approx(raw_per_year / 365.0, rel=1e-3)
    assert g.theta < 0.0


def test_gamma_and_vega_do_not_depend_on_the_option_kind():
    call = quant.greeks(S, K, T, r, SIGMA, Q, "call")
    put = quant.greeks(S, K, T, r, SIGMA, Q, "put")
    assert call.gamma == pytest.approx(put.gamma, rel=1e-12)
    assert call.vega == pytest.approx(put.vega, rel=1e-12)


def test_delta_bounds_and_signs():
    call = quant.greeks(S, K, T, r, SIGMA, Q, "call")
    put = quant.greeks(S, K, T, r, SIGMA, Q, "put")
    assert 0.0 < call.delta < 1.0
    assert -1.0 < put.delta < 0.0
    assert call.rho > 0.0 and put.rho < 0.0


def test_greeks_at_expiry_do_not_explode():
    g = quant.greeks(S=120, K=100, T=0.0, r=0.03, sigma=0.2, kind="call")
    assert g.delta == 1.0
    assert g.gamma == 0.0 and g.vega == 0.0


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.05, 0.15, 0.3, 0.75, 1.5])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_inverts_the_price(sigma, kind):
    price = quant.black_scholes(S=100, K=110, T=0.8, r=0.02, sigma=sigma, kind=kind)
    recovered = quant.implied_vol(price, S=100, K=110, T=0.8, r=0.02, kind=kind)
    assert recovered == pytest.approx(sigma, abs=1e-5)


def test_implied_vol_rejects_a_price_below_intrinsic():
    with pytest.raises(ValueError, match="intrinsic"):
        quant.implied_vol(price=0.5, S=150, K=100, T=1.0, r=0.0)


def test_implied_vol_rejects_expiry():
    with pytest.raises(ValueError, match="expiry"):
        quant.implied_vol(price=5.0, S=100, K=100, T=0.0, r=0.0)


# ---------------------------------------------------------------------------
# Paths and portfolio risk
# ---------------------------------------------------------------------------


np = pytest.importorskip("numpy")


def test_gbm_shape_start_and_drift():
    paths = quant.gbm_paths(S0=100.0, mu=0.06, sigma=0.2, T=1.0, steps=252,
                            trials=40_000, seed=99)
    assert paths.shape == (40_000, 253)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0.0)          # a lognormal price can never go negative
    # E[S_T] = S0 · exp(mu·T) for this parameterisation.
    assert float(paths[:, -1].mean()) == pytest.approx(100.0 * math.exp(0.06), rel=0.02)


def test_gbm_is_reproducible():
    a = quant.gbm_paths(S0=100, mu=0.05, sigma=0.2, T=1, steps=50, trials=100, seed=3)
    b = quant.gbm_paths(S0=100, mu=0.05, sigma=0.2, T=1, steps=50, trials=100, seed=3)
    assert np.array_equal(a, b)


def test_jump_diffusion_keeps_its_drift_and_fattens_the_tails():
    common = dict(S0=100.0, mu=0.05, T=1.0, steps=252, trials=60_000, seed=11)
    smooth = quant.gbm_paths(sigma=0.2, **common)
    jumpy = quant.merton_jump_paths(
        sigma=0.2, jump_intensity=1.5, jump_mean=-0.08, jump_sd=0.15, **common
    )
    # The compensator exists so the drift survives the jumps.
    assert float(jumpy[:, -1].mean()) == pytest.approx(100.0 * math.exp(0.05), rel=0.04)
    # But the distribution should be wider than plain GBM.
    assert float(jumpy[:, -1].std()) > float(smooth[:, -1].std())


def test_var_is_a_positive_loss_and_grows_with_confidence():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0004, 0.012, 250_000)
    assert quant.historical_var(returns, 0.99) > 0.0
    assert quant.historical_var(returns, 0.999) > quant.historical_var(returns, 0.99)


def test_parametric_var_matches_historical_for_normal_returns():
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0, 0.01, 400_000)
    historical = quant.historical_var(returns, 0.99)
    parametric = quant.parametric_var(returns, 0.99)
    assert parametric == pytest.approx(historical, rel=0.03)


def test_parametric_var_accepts_explicit_moments():
    # 99% one-tailed normal quantile is 2.326348.
    var = quant.parametric_var(level=0.99, mean=0.0, sd=1.0)
    assert var == pytest.approx(2.326348, abs=1e-4)


def test_expected_shortfall_exceeds_var():
    rng = np.random.default_rng(7)
    returns = rng.standard_t(4, 200_000) * 0.01
    assert quant.expected_shortfall(returns, 0.99) > quant.historical_var(returns, 0.99)


# ---------------------------------------------------------------------------
# Heston
# ---------------------------------------------------------------------------


def test_heston_collapses_to_black_scholes_as_vol_of_vol_vanishes():
    """The strongest available check on the transform code.

    With xi -> 0 and v0 = theta, variance is deterministic and Heston must
    reproduce Black-Scholes at sigma = sqrt(theta). If the characteristic
    function or the inversion has a sign error, this is where it shows.
    """
    bs = quant.black_scholes(S=100, K=100, T=1.0, r=0.03, sigma=0.2)
    heston = quant.heston_price(
        S=100, K=100, T=1.0, r=0.03,
        v0=0.04, kappa=2.0, theta=0.04, xi=1e-6, rho=0.0,
    )
    assert heston == pytest.approx(bs, abs=1e-3)


# These parameters violate Feller on purpose; the warning itself is asserted by
# test_heston_warns_when_feller_is_violated, so here it is only noise.
@pytest.mark.filterwarnings("ignore:Feller condition violated:RuntimeWarning")
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_heston_respects_put_call_parity(strike):
    params = dict(S=100.0, K=strike, T=0.9, r=0.025,
                  v0=0.05, kappa=1.4, theta=0.045, xi=0.45, rho=-0.65)
    call = quant.heston_price(kind="call", **params)
    put = quant.heston_price(kind="put", **params)
    assert call - put == pytest.approx(100.0 - strike * math.exp(-0.025 * 0.9), abs=1e-6)


def test_heston_warns_when_feller_is_violated():
    with pytest.warns(RuntimeWarning, match="Feller"):
        quant.heston_price(S=100, K=100, T=1.0, r=0.02,
                           v0=0.04, kappa=0.5, theta=0.04, xi=0.9, rho=-0.5)
