"""Tests for the credit-risk layer.

Structural (Merton), reduced-form (hazard rates and CDS), portfolio (ASRF /
Vasicek, Basel IRB, one-factor copula simulation) and ratings migration.
Identities first, published limits second, magic numbers last.
"""

from __future__ import annotations

import math

import pytest

from riskpy import credit, quant
from riskpy._special import norm_cdf, norm_ppf


# ---------------------------------------------------------------------------
# Expected and unexpected loss
# ---------------------------------------------------------------------------


def test_expected_and_unexpected_loss():
    assert credit.expected_loss(0.02, 0.45, 1_000_000.0) == pytest.approx(9_000.0)
    assert credit.unexpected_loss(0.02, 0.45, 1_000_000.0) == pytest.approx(math.sqrt(0.02 * 0.98) * 0.45 * 1e6)
    assert credit.unexpected_loss(0.0, 0.45, 1e6) == 0.0
    with pytest.raises(ValueError):
        credit.expected_loss(1.5, 0.45, 1e6)
    with pytest.raises(ValueError):
        credit.expected_loss(0.02, -0.1, 1e6)


# ---------------------------------------------------------------------------
# Merton
# ---------------------------------------------------------------------------


def test_merton_is_a_call_on_the_assets():
    m = credit.merton(assets=120.0, debt=100.0, T=2.0, r=0.03, sigma=0.25)
    assert m.equity == pytest.approx(quant.black_scholes(120.0, 100.0, 2.0, 0.03, 0.25, kind="call"), abs=1e-12)
    assert m.equity + m.debt_value == pytest.approx(120.0, abs=1e-12)
    assert m.pd == pytest.approx(norm_cdf(-m.distance_to_default), abs=1e-14)
    assert 0.0 < m.pd < 0.5
    assert m.credit_spread > 0.0
    # Leverage is the *discounted* face over assets, the quantity the
    # distance-to-default is measured against.
    assert m.leverage == pytest.approx(100.0 * math.exp(-0.03 * 2.0) / 120.0)


def test_merton_spread_vanishes_without_volatility():
    m = credit.merton(assets=150.0, debt=100.0, T=1.0, r=0.03, sigma=1e-9)
    assert m.credit_spread == pytest.approx(0.0, abs=1e-8)
    assert m.pd == pytest.approx(0.0, abs=1e-8)


def test_merton_physical_measure_uses_the_drift():
    neutral = credit.merton(assets=120.0, debt=100.0, T=2.0, r=0.03, sigma=0.25)
    physical = credit.merton(assets=120.0, debt=100.0, T=2.0, r=0.03, sigma=0.25, mu=0.10)
    assert physical.distance_to_default > neutral.distance_to_default
    assert physical.pd < neutral.pd
    # Pricing quantities do not depend on the physical drift.
    assert physical.equity == pytest.approx(neutral.equity)


def test_merton_validation():
    with pytest.raises(ValueError):
        credit.merton(assets=-1.0, debt=100.0, T=1.0, r=0.03, sigma=0.2)
    with pytest.raises(ValueError):
        credit.merton(assets=100.0, debt=100.0, T=0.0, r=0.03, sigma=0.2)
    with pytest.raises(ValueError):
        credit.merton(assets=100.0, debt=100.0, T=1.0, r=0.03, sigma=-0.2)


# ---------------------------------------------------------------------------
# Reduced form
# ---------------------------------------------------------------------------


def test_survival_probability_and_credit_triangle():
    assert credit.survival_probability(0.02, 5.0) == pytest.approx(math.exp(-0.10))
    assert credit.survival_probability(0.02, 0.0) == 1.0
    h = credit.hazard_from_spread(0.012, 0.4)
    assert h == pytest.approx(0.012 / 0.6)
    assert credit.spread_from_hazard(h, 0.4) == pytest.approx(0.012)
    with pytest.raises(ValueError):
        credit.hazard_from_spread(0.01, 1.0)


def test_cds_par_spread_approximates_the_credit_triangle():
    hazard, recovery = 0.01, 0.4
    spread = credit.cds_par_spread(hazard, recovery, r=0.03, maturity=5.0, frequency=4)
    assert spread == pytest.approx(hazard * (1.0 - recovery), rel=0.01)
    legs = credit.cds_legs(hazard, recovery, r=0.03, maturity=5.0, frequency=4)
    assert hasattr(legs, "premium") or hasattr(legs, "premium_leg")


def test_cds_spread_is_increasing_in_hazard_and_decreasing_in_recovery():
    low = credit.cds_par_spread(0.005, 0.4, 0.03, 5.0)
    high = credit.cds_par_spread(0.02, 0.4, 0.03, 5.0)
    assert high > low
    assert credit.cds_par_spread(0.01, 0.6, 0.03, 5.0) < credit.cds_par_spread(0.01, 0.2, 0.03, 5.0)
    with pytest.raises(ValueError):
        credit.cds_par_spread(-0.01, 0.4, 0.03, 5.0)


# ---------------------------------------------------------------------------
# ASRF / Vasicek and Basel
# ---------------------------------------------------------------------------


def test_asrf_conditional_pd_properties():
    assert credit.asrf_conditional_pd(0.02, 0.0, 0.999) == pytest.approx(0.02, abs=1e-15)
    a = credit.asrf_conditional_pd(0.02, 0.2, 0.99)
    b = credit.asrf_conditional_pd(0.02, 0.2, 0.999)
    c = credit.asrf_conditional_pd(0.02, 0.4, 0.999)
    assert 0.02 < a < b < c < 1.0
    expected = norm_cdf((norm_ppf(0.02) + math.sqrt(0.2) * norm_ppf(0.999)) / math.sqrt(0.8))
    assert b == pytest.approx(expected, abs=1e-14)
    with pytest.raises(ValueError):
        credit.asrf_conditional_pd(0.02, 1.5, 0.999)
    with pytest.raises(ValueError):
        credit.asrf_conditional_pd(0.02, 0.2, 1.0)


def test_vasicek_loss_distribution():
    pd, rho = 0.03, 0.15
    quantile = credit.vasicek_loss_quantile(0.999, pd, rho)
    assert quantile == pytest.approx(credit.asrf_conditional_pd(pd, rho, 0.999), abs=1e-14)
    assert credit.vasicek_loss_cdf(quantile, pd, rho) == pytest.approx(0.999, abs=1e-12)
    assert credit.vasicek_loss_cdf(0.0, pd, rho) == 0.0
    assert credit.vasicek_loss_cdf(1.0, pd, rho) == 1.0
    # The median loss is below the mean: the distribution is right-skewed.
    assert credit.vasicek_loss_quantile(0.5, pd, rho) < pd


def test_basel_correlation_limits_and_classes():
    assert credit.basel_correlation(1e-12) == pytest.approx(0.24, abs=1e-9)
    assert credit.basel_correlation(1.0) == pytest.approx(0.12, abs=1e-9)
    assert credit.basel_correlation(0.01, "retail_mortgage") == 0.15
    assert credit.basel_correlation(0.01, "retail_revolving") == 0.04
    other = credit.basel_correlation(0.01, "retail_other")
    assert 0.03 < other < 0.16
    with pytest.raises(ValueError):
        credit.basel_correlation(0.01, "crypto")


def test_basel_irb_capital_reproduces_the_formula():
    pd, lgd, ead, m = 0.01, 0.45, 1_000_000.0, 2.5
    rho = credit.basel_correlation(pd)
    cond = norm_cdf((norm_ppf(pd) + math.sqrt(rho) * norm_ppf(0.999)) / math.sqrt(1.0 - rho))
    b = (0.11852 - 0.05478 * math.log(pd)) ** 2
    k = (lgd * cond - pd * lgd) * (1.0 + (m - 2.5) * b) / (1.0 - 1.5 * b)
    out = credit.basel_irb_capital(pd, lgd, ead, maturity=m)
    assert out.risk_weight == pytest.approx(12.5 * k, rel=1e-12)
    # BCBS, "An Explanatory Note on the Basel II IRB Risk Weight Functions" (2005),
    # corporate table: PD 1.00%, LGD 45%, M 2.5 -> risk weight 92.32%.
    assert out.risk_weight == pytest.approx(0.9232, abs=5e-4)
    longer = credit.basel_irb_capital(pd, lgd, ead, maturity=5.0)
    assert longer.risk_weight > out.risk_weight
    with pytest.raises(ValueError):
        credit.basel_irb_capital(0.0, lgd, ead)


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------


def test_portfolio_loss_simulation():
    np = pytest.importorskip("numpy")
    from riskpy.mc import Result

    pds = [0.01, 0.02, 0.05, 0.10]
    lgds = [0.4, 0.45, 0.5, 0.6]
    eads = [1e6, 2e6, 5e5, 3e5]
    expected = sum(p * l * e for p, l, e in zip(pds, lgds, eads))
    independent = credit.credit_portfolio_loss(pds, lgds, eads, rho=0.0, trials=60_000, seed=1)
    assert isinstance(independent, Result)
    assert independent.label == "portfolio loss"
    assert independent.mean == pytest.approx(expected, rel=0.03)
    clustered = credit.credit_portfolio_loss(pds, lgds, eads, rho=0.9, trials=60_000, seed=1)
    assert clustered.mean == pytest.approx(expected, rel=0.05)
    assert clustered.var(0.999) > independent.var(0.999)
    again = credit.credit_portfolio_loss(pds, lgds, eads, rho=0.9, trials=60_000, seed=1)
    assert np.array_equal(again.values, clustered.values)
    per_exposure = credit.credit_portfolio_loss(pds, lgds, eads, rho=[0.1, 0.2, 0.3, 0.4], trials=2_000, seed=1)
    assert per_exposure.trials == 2_000
    with pytest.raises(ValueError):
        credit.credit_portfolio_loss(pds, lgds[:2], eads, rho=0.2)
    with pytest.raises(ValueError):
        credit.credit_portfolio_loss(pds, lgds, eads, rho=1.5)


# ---------------------------------------------------------------------------
# Rating transitions
# ---------------------------------------------------------------------------


def test_transition_matrix_powers_and_pds():
    tm = credit.sp_transition_matrix()
    assert tm.ratings[-1] == "D"
    assert tm.probability("D", "D") == 1.0
    two = tm.power(2)
    for rating in tm.ratings:
        assert sum(two.probability(rating, to) for to in tm.ratings) == pytest.approx(1.0, abs=1e-10)
    cumulative = [tm.cumulative_pd("BBB", n) for n in range(1, 8)]
    assert all(b >= a for a, b in zip(cumulative, cumulative[1:]))
    assert tm.cumulative_pd("AAA", 1) < tm.cumulative_pd("B", 1)
    assert tm.marginal_pd("BBB", 1) == pytest.approx(tm.cumulative_pd("BBB", 1))
    assert tm.marginal_pd("BBB", 3) == pytest.approx(tm.cumulative_pd("BBB", 3) - tm.cumulative_pd("BBB", 2), abs=1e-12)
    assert tm.power(1).matrix == tm.matrix


def test_transition_matrix_validation():
    with pytest.raises(ValueError):
        credit.TransitionMatrix([[0.9, 0.2], [0.0, 1.0]], ["A", "D"])  # rows must sum to 1
    with pytest.raises(ValueError):
        credit.TransitionMatrix([[0.9, 0.1], [0.5, 0.5]], ["A", "D"])  # default must absorb
    with pytest.raises(ValueError):
        credit.TransitionMatrix([[0.9, 0.1], [0.0, 1.0]], ["A", "B"])  # last state must be D
    with pytest.raises(ValueError):
        credit.TransitionMatrix([[0.9, 0.1]], ["A", "D"])  # not square
    tm = credit.sp_transition_matrix()
    with pytest.raises(ValueError):
        tm.cumulative_pd("ZZZ", 1)
    assert tm.power(0).probability("AAA", "AAA") == 1.0  # the identity
    with pytest.raises(ValueError):
        tm.power(-1)


def test_verification_checks_all_pass():
    for name, value, reference, tolerance in credit._verification_checks():
        assert abs(value - reference) <= tolerance, name
