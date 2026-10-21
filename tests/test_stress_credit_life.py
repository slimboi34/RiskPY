"""Regression and stress tests for the credit and life fixes.

Each regression test pins one reproduced bug; the randomized tests (fixed seeds)
compare against independent references — scipy's normal distribution for
Basel and Merton, scipy.integrate.quad for the CDS legs, exact rational
arithmetic for annuities near i = 0, and the closed-table identities for life
contingencies.
"""

from __future__ import annotations

import math
import random
import warnings
from fractions import Fraction

import pytest

from riskpy import credit, life
from riskpy.errors import InputError
from riskpy.life import LifeTable


# ---------------------------------------------------------------------------
# 1. CDS legs: exact integration of protection and accrued premium
# ---------------------------------------------------------------------------


def _schedule(maturity, frequency):
    periods = int(math.ceil(maturity * frequency - 1e-9))
    return [maturity - (periods - k) / frequency for k in range(1, periods + 1)]


def _cds_reference(knots, recovery, r, maturity, frequency):
    """Par spread by numerical quadrature of the continuous-default legs."""
    integrate = pytest.importorskip("scipy.integrate")
    times = [t for t, _ in knots[:-1]]

    def hazard(u):
        for t, h in knots:
            if u < t:
                return h
        return knots[-1][1]

    def cumulative(u):
        total, start = 0.0, 0.0
        for t, h in knots:
            end = min(u, t)
            if end > start:
                total += h * (end - start)
            start = t
            if t >= u:
                break
        return total

    def density(u):
        return hazard(u) * math.exp(-cumulative(u) - r * u)

    annuity = protection = 0.0
    start = 0.0
    for end in _schedule(maturity, frequency):
        pts = [t for t in times if start < t < end] or None
        annuity += (end - start) * math.exp(-cumulative(end) - r * end)
        annuity += integrate.quad(lambda u: (u - start) * density(u), start, end, points=pts,
                                  epsabs=0.0, epsrel=1e-13, limit=200)[0]
        protection += (1.0 - recovery) * integrate.quad(density, start, end, points=pts,
                                                        epsabs=0.0, epsrel=1e-13, limit=200)[0]
        start = end
    return protection / annuity


def test_cds_par_spread_at_high_hazard_is_exact():
    # The midpoint rule gave 2.667 and 4.80 here.
    assert credit.cds_par_spread(5.0, 0.4, 0.03, 5.0) == pytest.approx(3.0089788704, rel=1e-9)
    assert credit.cds_par_spread(50.0, 0.4, 0.0, 5.0) == pytest.approx(30.0, rel=1e-13)


def test_cds_par_spread_unchanged_at_low_hazard():
    assert credit.cds_par_spread(0.05, 0.4, 0.03, 5.0) == pytest.approx(3.0113e-2, rel=2e-5)
    assert credit.cds_par_spread(0.01, 0.4, 0.03, 5.0) == pytest.approx(0.006 * 1.00376, rel=1e-4)


@pytest.mark.parametrize("frequency", [1, 2, 4, 12])
@pytest.mark.parametrize("hazard", [1e-4, 0.01, 0.3, 4.0, 50.0])
def test_cds_reproduces_the_credit_triangle_exactly_at_zero_rates(hazard, frequency):
    assert credit.cds_par_spread(hazard, 0.4, 0.0, 5.0, frequency) == pytest.approx(0.6 * hazard, rel=1e-12)


def test_cds_matches_quadrature_across_hazards():
    rng = random.Random(20261021)
    for _ in range(40):
        hazard = 10 ** rng.uniform(-4, math.log10(50))
        recovery = rng.uniform(0.0, 0.8)
        r = rng.uniform(-0.02, 0.10)
        maturity = rng.choice([0.6, 1.0, 2.6, 5.0])
        frequency = rng.choice([1, 2, 4, 12])
        got = credit.cds_par_spread(hazard, recovery, r, maturity, frequency)
        ref = _cds_reference([(math.inf, hazard)], recovery, r, maturity, frequency)
        assert got == pytest.approx(ref, rel=1e-9), (hazard, recovery, r, maturity, frequency)


def test_cds_piecewise_hazard_matches_quadrature():
    rng = random.Random(7)
    for _ in range(12):
        t1 = rng.uniform(0.1, 2.0)
        t2 = t1 + rng.uniform(0.1, 3.0)
        h = [10 ** rng.uniform(-4, 1.5) for _ in range(3)]
        curve = [(t1, h[0]), (t2, h[1]), (t2 + 1.0, h[2])]
        r = rng.uniform(0.0, 0.08)
        maturity = rng.choice([2.6, 5.0])
        frequency = rng.choice([2, 4])
        got = credit.cds_legs(curve, 0.4, r, maturity, frequency=frequency)
        ref = _cds_reference([(t1, h[0]), (t2, h[1]), (math.inf, h[2])], 0.4, r, maturity, frequency)
        assert got.par_spread == pytest.approx(ref, rel=1e-9)
        assert got.upfront == pytest.approx(0.0, abs=1e-12)


def test_hazard_from_spread_gap_matches_its_docstring():
    # About 0.4% at r = 3% quarterly, for hazards up to ~20%.
    for h in (0.001, 0.01, 0.05, 0.2):
        gap = credit.cds_par_spread(h, 0.4, 0.03, 5.0) / credit.spread_from_hazard(h, 0.4) - 1.0
        assert 0.0035 < gap < 0.0040


def test_hazard_string_is_a_clear_error():
    with pytest.raises(InputError, match="string") as info:
        credit.cds_par_spread("0.02", 0.4, 0.0, 5.0)
    assert info.value.code == "RP101"
    with pytest.raises(InputError):
        credit.survival_probability("0.02", 1.0)


# ---------------------------------------------------------------------------
# 2. Merton: debt priced directly, spread from logs
# ---------------------------------------------------------------------------


def _merton_reference(V, D, T, r, s):
    special = pytest.importorskip("scipy.special")
    vol = s * math.sqrt(T)
    d1 = (math.log(V / D) + (r + 0.5 * s * s) * T) / vol
    d2 = d1 - vol
    log_b = max(a := math.log(D) - r * T + float(special.log_ndtr(d2)),
                b := math.log(V) + float(special.log_ndtr(-d1)))
    log_b += math.log1p(math.exp(min(a, b) - log_b))
    spread = -(log_b - math.log(D)) / T - r
    return spread, d2


@pytest.mark.parametrize("args", [
    (100, 100, 100, 0.05, 10),
    (100, 100, 1e6, 0.05, 0.2),
    (100, 100, 10, 50, 0.2),
])
def test_merton_extremes_no_longer_crash(args):
    m = credit.merton(*args)
    assert math.isfinite(m.credit_spread) and m.credit_spread >= 0.0
    assert m.debt_value >= 0.0


def test_merton_spread_values():
    assert credit.merton(1e12, 1, 1, 0.05, 0.2).credit_spread == pytest.approx(0.0, abs=1e-15)
    assert credit.merton(100, 100, 10, 0.05, 5).credit_spread == pytest.approx(3.33093, abs=1e-5)
    ref, _ = _merton_reference(100, 100, 100, 0.05, 10)
    assert credit.merton(100, 100, 100, 0.05, 10).credit_spread == pytest.approx(ref, rel=1e-10)


def test_merton_matches_scipy_reference():
    norm = pytest.importorskip("scipy.stats").norm
    rng = random.Random(1974)
    for _ in range(300):
        V = 10 ** rng.uniform(1, 3)
        D = 10 ** rng.uniform(1, 3)
        T = 10 ** rng.uniform(-1, 1.5)
        r = rng.uniform(-0.02, 0.12)
        s = 10 ** rng.uniform(-1.3, 0.7)
        m = credit.merton(V, D, T, r, s)
        spread, d2 = _merton_reference(V, D, T, r, s)
        assert m.credit_spread == pytest.approx(spread, rel=1e-8, abs=1e-12)
        assert m.distance_to_default == pytest.approx(d2, rel=1e-12, abs=1e-12)
        assert m.pd == pytest.approx(float(norm.cdf(-d2)), rel=1e-9, abs=1e-300)
        assert m.equity + m.debt_value == pytest.approx(V, rel=1e-11)
        assert m.credit_spread >= -1e-15


def test_merton_spread_is_tiny_but_positive_for_a_safe_firm():
    # assets − equity cancelled to a negative spread here (−3e-8).
    m = credit.merton(1e9, 1, 1, 0.05, 0.2)
    assert m.credit_spread == pytest.approx(0.0, abs=1e-15)
    assert m.debt_value == pytest.approx(math.exp(-0.05), rel=1e-15)


# ---------------------------------------------------------------------------
# 3. Basel IRB
# ---------------------------------------------------------------------------


def _basel_reference(pd, lgd, M, asset_class):
    norm = pytest.importorskip("scipy.stats").norm
    if asset_class in ("corporate", "sovereign", "bank"):
        w = (1 - math.exp(-50 * pd)) / (1 - math.exp(-50))
        R = 0.12 * w + 0.24 * (1 - w)
    elif asset_class == "retail_mortgage":
        R = 0.15
    elif asset_class == "retail_revolving":
        R = 0.04
    else:
        w = (1 - math.exp(-35 * pd)) / (1 - math.exp(-35))
        R = 0.03 * w + 0.16 * (1 - w)
    K = lgd * norm.cdf((norm.ppf(pd) + math.sqrt(R) * norm.ppf(0.999)) / math.sqrt(1 - R)) - pd * lgd
    if asset_class in ("corporate", "sovereign", "bank"):
        b = (0.11852 - 0.05478 * math.log(pd)) ** 2
        K *= (1 + (M - 2.5) * b) / (1 - 1.5 * b)
    return float(K)


def test_basel_matches_the_bcbs_formula():
    rng = random.Random(2005)
    classes = ["corporate", "sovereign", "bank", "retail_mortgage", "retail_revolving", "retail_other"]
    for _ in range(400):
        pd = 10 ** rng.uniform(-5, -0.01)
        lgd = rng.uniform(0.0, 1.0)
        M = rng.uniform(1.0, 5.0)
        asset_class = rng.choice(classes)
        got = credit.basel_irb_capital(pd, lgd, 1.0, M, asset_class).capital_requirement
        assert got == pytest.approx(_basel_reference(pd, lgd, M, asset_class), rel=1e-10, abs=1e-15)


@pytest.mark.parametrize("pd", [2.0e-6, 2.93e-6, 3e-6, 9e-6])
def test_basel_refuses_pds_where_the_maturity_adjustment_breaks(pd):
    with pytest.raises(InputError, match="maturity adjustment") as info:
        credit.basel_irb_capital(pd, 0.45, 1.0)
    assert info.value.param == "pd"
    # Retail has no maturity adjustment, so tiny PDs stay valid there.
    assert credit.basel_irb_capital(pd, 0.45, 1.0, asset_class="retail_other").capital_requirement >= 0.0


@pytest.mark.parametrize("maturity", [1.0, 2.5, 5.0])
def test_basel_capital_is_increasing_in_pd_above_the_minimum(maturity):
    pds = [10 ** (-5 + 3 * k / 300) for k in range(301)]
    ks = [credit.basel_irb_capital(p, 0.45, 1.0, maturity).capital_requirement for p in pds]
    assert all(b > a for a, b in zip(ks, ks[1:]))
    assert all(k > 0.0 for k in ks)


# ---------------------------------------------------------------------------
# 4. Annuities and nominal rates near i = 0
# ---------------------------------------------------------------------------


def _exact_annuity(i, n, due=False, accumulate=False):
    v = 1 / (1 + Fraction(i))
    total = sum(v ** (k if due else k + 1) for k in range(n))
    if accumulate:
        total *= (1 + Fraction(i)) ** n
    return float(total)


@pytest.mark.parametrize("i", [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, -1e-12, -1e-6, 0.05])
@pytest.mark.parametrize("n", [1, 7, 30])
def test_annuities_are_accurate_near_zero_interest(i, n):
    assert life.annuity_certain(i, n) == pytest.approx(_exact_annuity(i, n), rel=1e-14)
    assert life.annuity_certain(i, n, due=True) == pytest.approx(_exact_annuity(i, n, due=True), rel=1e-14)
    assert life.accumulated_annuity(i, n) == pytest.approx(_exact_annuity(i, n, accumulate=True), rel=1e-14)
    assert life.accumulated_annuity(i, n, due=True) == pytest.approx(
        _exact_annuity(i, n, due=True, accumulate=True), rel=1e-14)


def test_annuity_identities_near_zero():
    rng = random.Random(11)
    for _ in range(200):
        i = rng.choice([1, -1]) * 10 ** rng.uniform(-15, -2)
        n = rng.randint(0, 60)
        a = life.annuity_certain(i, n)
        assert life.annuity_certain(i, n, due=True) == pytest.approx((1 + i) * a, rel=1e-14)
        assert life.accumulated_annuity(i, n) == pytest.approx(a * (1 + i) ** n, rel=1e-13)
        # a_n -> n as i -> 0, with first-order term -n(n+1)/2 · i.
        assert a == pytest.approx(n - n * (n + 1) / 2 * i, rel=1e-12 + abs(i) * n * n)


@pytest.mark.parametrize("i", [1e-12, 1e-9, 1e-6, -1e-9])
@pytest.mark.parametrize("m", [2, 4, 12, 365])
def test_nominal_rate_near_zero(i, m):
    exact = m * math.expm1(math.log1p(i) / m)
    assert life.nominal_rate(i, m) == pytest.approx(exact, rel=1e-15)
    assert life.nominal_rate(i, m) == pytest.approx(i - (m - 1) / (2 * m) * i * i, rel=1e-12)
    assert life.effective_from_nominal(life.nominal_rate(i, m), m) == pytest.approx(i, rel=1e-6)


def test_annuity_overflow_is_a_structured_error():
    with pytest.raises(InputError) as info:
        life.annuity_certain(-0.5, 2000)
    assert info.value.code == "RP304"
    with pytest.raises(InputError):
        life.accumulated_annuity(1.0, 2000)
    assert life.annuity_certain(0.05, 10**6) == pytest.approx(20.0, rel=1e-14)


# ---------------------------------------------------------------------------
# 5. Transition matrices
# ---------------------------------------------------------------------------


def test_large_transition_powers_do_not_trip_validation():
    tm = credit.sp_transition_matrix()
    p = tm.power(3242)
    for row in p.matrix:
        assert all(0.0 <= v <= 1.0 for v in row)
        assert math.fsum(row) == pytest.approx(1.0, abs=1e-12)
    assert tm.cumulative_pd("AAA", 10**6) == pytest.approx(1.0, abs=1e-12)
    assert tm.power(4096).probability("D", "D") == 1.0


def test_rows_within_tolerance_are_renormalised():
    tm = credit.TransitionMatrix([[0.5, 0.5 + 5e-9], [0.0, 1.0]], ["A", "D"])
    assert math.fsum(tm.matrix[0]) == pytest.approx(1.0, abs=1e-15)
    assert math.fsum(tm.power(2000).matrix[0]) == pytest.approx(1.0, abs=1e-12)
    near = credit.TransitionMatrix([[0.9, 0.1], [1e-13, 1 - 1e-13]], ["A", "D"])
    assert near.matrix[-1] == (0.0, 1.0)


def test_sp_matrix_entries_survive_construction_exactly():
    tm = credit.sp_transition_matrix()
    assert tm.probability("BBB", "D") == 0.0021
    assert tm.probability("CCC/C", "D") == 0.3514
    assert tm.power(1).matrix == tm.matrix


def test_transition_label_lookup_converts_like_construction():
    tm = credit.TransitionMatrix([[0.9, 0.1], [0.0, 1.0]], [1, "D"])
    assert tm.ratings == ("1", "D")
    assert tm.cumulative_pd(1, 2) == pytest.approx(0.19)
    assert tm.probability("1", "D") == 0.1


def test_check_count_rejects_bools():
    tm = credit.sp_transition_matrix()
    with pytest.raises(InputError):
        tm.power(True)
    with pytest.raises(InputError):
        tm.cumulative_pd("BBB", False)
    with pytest.raises(InputError):
        tm.power(float("inf"))
    assert tm.cumulative_pd("BBB", 2.0) == tm.cumulative_pd("BBB", 2)


# ---------------------------------------------------------------------------
# 6-8. Life tables: laws, from_lx, validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,c,max_age", [(1e-5, 1.5, 2000), (1e-5, 1e10, 130), (1e-3, 3.0, 800)])
def test_steep_laws_close_early_instead_of_overflowing(B, c, max_age):
    with pytest.warns(RuntimeWarning, match="certainty"):
        table = LifeTable.gompertz(B, c, start_age=0, max_age=max_age)
    assert table.qx(table.omega - 1) == 1.0
    assert table.lx(table.omega) == 0.0
    with pytest.warns(RuntimeWarning, match="certainty"):
        LifeTable.makeham(0.001, B, c, start_age=0, max_age=max_age)


def test_makeham_still_matches_sult_values():
    sult = LifeTable.sult()
    assert sult.lx(100) == pytest.approx(6248.17, abs=0.01)
    assert life.whole_life_annuity_due(sult, 40, 0.05) == pytest.approx(18.457757, abs=1e-6)


def test_from_lx_accepts_survival_below_double_epsilon():
    table = LifeTable.from_lx([1.0, 1e-20, 0.0])
    assert table.omega == 2
    assert table.lx(1) == 1e-20
    assert table.qx(0) < 1.0 and table.qx(1) == 1.0
    assert table.tpx(1, 0) == 1e-20
    long = LifeTable.from_lx([1.0] + [10.0 ** (-17 * k) for k in range(1, 18)] + [0.0])
    assert long.omega == 18 and long.lx(17) == 1e-289


@pytest.mark.parametrize("bad", ["0.5", None, 1.0, b"0.5"])
def test_life_table_rejects_non_sequences(bad):
    with pytest.raises(InputError, match="sequence") as info:
        LifeTable(bad)
    assert info.value.code == "RP501"
    with pytest.raises(InputError, match="sequence"):
        LifeTable.from_lx(bad)


@pytest.mark.parametrize("radix", [1e308, 1e301, 1e-320])
def test_radix_out_of_range_is_rejected(radix):
    with pytest.raises(InputError, match="radix"):
        LifeTable([0.5, 1.0], radix=radix)


def test_joint_life_is_radix_invariant():
    q = [0.01 * 1.1 ** k for k in range(40)] + [1.0]
    small, big = LifeTable(q, radix=1.0), LifeTable(q, radix=1e200)
    a = life.joint_life_annuity_due(small, 0, 5, 0.04)
    assert life.joint_life_annuity_due(big, 0, 5, 0.04) == pytest.approx(a, rel=1e-13)
    assert life.joint_life_insurance(big, 0, 5, 0.04) == pytest.approx(
        life.joint_life_insurance(small, 0, 5, 0.04), rel=1e-13)
    assert life.whole_life_annuity_due(big, 0, 0.04) == pytest.approx(
        life.whole_life_annuity_due(small, 0, 0.04), rel=1e-13)


def test_numpy_scalar_ages_are_accepted():
    np = pytest.importorskip("numpy")
    table = LifeTable([0.1, 0.2, 1.0])
    for age in (np.float32(1.0), np.float64(1.0), np.int32(1), Fraction(1)):
        assert table.qx(age) == 0.2
    with pytest.raises(InputError):
        table.qx(np.float32(0.5))
    with pytest.raises(InputError):
        table.qx(True)


def test_life_identities_over_random_laws():
    rng = random.Random(1825)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tables = []
        for _ in range(16):
            B = 10 ** rng.uniform(-6, -3)
            c = rng.uniform(1.05, 1.2)
            start = rng.randint(0, 60)
            if rng.random() < 0.5:
                tables.append(LifeTable.gompertz(B, c, start_age=start, max_age=start + 90))
            else:
                tables.append(LifeTable.makeham(rng.uniform(0, 0.003), B, c, start_age=start, max_age=start + 90))
    for table in tables:
        for _ in range(4):
            i = rng.choice([0.0, 0.03, 0.06, rng.uniform(-0.05, 0.15)])
            d = i / (1 + i)
            x = rng.choice(table.ages)
            n = rng.randint(1, table.omega - x)
            A = life.whole_life_insurance(table, x, i)
            a = life.whole_life_annuity_due(table, x, i)
            assert A == pytest.approx(1 - d * a, abs=1e-12)
            assert life.term_insurance(table, x, n, i) + life.deferred_whole_life_insurance(table, x, n, i) == \
                pytest.approx(A, abs=1e-12)
            assert life.temporary_annuity_due(table, x, n, i) + life.deferred_annuity_due(table, x, n, i) == \
                pytest.approx(a, rel=1e-12)
            assert life.endowment_insurance(table, x, n, i) == pytest.approx(
                1 - d * life.temporary_annuity_due(table, x, n, i), abs=1e-12)
            assert sum(table.deferred_qx(t, x) for t in range(table.omega - x)) == pytest.approx(1.0, abs=1e-12)
            comm = life.Commutation(table, i)
            assert comm.M(x) / comm.D(x) == pytest.approx(A, rel=1e-10)
            assert comm.N(x) / comm.D(x) == pytest.approx(a, rel=1e-10)
            y = rng.choice(table.ages)
            assert life.joint_life_insurance(table, x, y, i) == pytest.approx(
                1 - d * life.joint_life_annuity_due(table, x, y, i), abs=1e-12)
            profile = life.reserve_profile(table, x, i, "endowment", n)
            P = life.endowment_premium(table, x, n, i)
            assert profile[0][1] == pytest.approx(0.0, abs=1e-12)
            for (t, V), (_, V_next) in zip(profile, profile[1:]):
                q = table.qx(x + t)
                assert (V + P) * (1 + i) == pytest.approx(q + (1 - q) * V_next, abs=1e-10)


# ---------------------------------------------------------------------------
# Portfolio simulation inputs
# ---------------------------------------------------------------------------


def test_portfolio_loss_accepts_a_zero_dimensional_rho():
    np = pytest.importorskip("numpy")
    args = ([0.02, 0.05, 0.1], [0.4, 0.5, 0.6], [1.0, 2.0, 3.0])
    a = credit.credit_portfolio_loss(*args, np.array(0.1), trials=2000, seed=3)
    b = credit.credit_portfolio_loss(*args, 0.1, trials=2000, seed=3)
    assert np.array_equal(a.values, b.values)
    c = credit.credit_portfolio_loss(*args, np.array([0.1, 0.1, 0.1]), trials=2000, seed=3)
    assert np.array_equal(a.values, c.values)
    with pytest.raises(InputError):
        credit.credit_portfolio_loss(*args, True, trials=10, seed=1)
    # A string is a scalar like every other numeric argument here, never
    # iterated character by character.
    d = credit.credit_portfolio_loss(*args, "0.1", trials=2000, seed=3)
    assert np.array_equal(a.values, d.values)


def test_portfolio_loss_mean_matches_expected_loss():
    np = pytest.importorskip("numpy")
    pds, lgds, eads = [0.02] * 40, [0.45] * 40, [1.0] * 40
    result = credit.credit_portfolio_loss(pds, lgds, eads, 0.2, trials=20000, seed=9)
    expected = 40 * 0.02 * 0.45
    assert float(np.mean(result.values)) == pytest.approx(expected, rel=0.05)
