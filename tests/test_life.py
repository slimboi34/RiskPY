"""Tests for the life contingencies layer.

Almost everything here is an identity — ``A_x = 1 − d·ä_x``, the endowment
decomposition, Fackler's reserve recursion, commutation ratios — because an
identity catches a whole class of errors where a single textbook number catches
a typo. The textbook numbers that are used (AMLCR's Standard Ultimate Life
Table) are cross-checked against the book's printed values *and* pinned to the
values this implementation produces, so a silent drift in either direction
shows up.
"""

from __future__ import annotations

import math
import warnings

import pytest

from riskpy import life
from riskpy.life import LifeTable

SULT = LifeTable.sult()
I = 0.05
V = 1.0 / (1.0 + I)
D = I / (1.0 + I)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_small_table_basics():
    table = LifeTable([0.1, 0.2, 0.5, 1.0], start_age=90, radix=1000.0)
    assert table.start_age == 90
    assert table.omega == 94
    assert len(table) == 4
    assert table.ages == [90, 91, 92, 93]
    assert table.lx(90) == 1000.0
    assert table.lx(91) == pytest.approx(900.0)
    assert table.lx(92) == pytest.approx(720.0)
    assert table.lx(93) == pytest.approx(360.0)
    assert table.lx(94) == 0.0
    assert table.qx(93) == 1.0 and table.px(93) == 0.0
    assert "90..93" in repr(table)


def test_deaths_are_consistent_with_survivors_and_rates():
    for x in SULT.ages:
        assert SULT.dx(x) == pytest.approx(SULT.lx(x) - SULT.lx(x + 1), rel=1e-12)
        assert SULT.dx(x) == pytest.approx(SULT.lx(x) * SULT.qx(x), rel=1e-9)
    assert sum(SULT.dx(x) for x in SULT.ages) == pytest.approx(SULT.radix, rel=1e-12)


def test_rates_must_be_probabilities():
    with pytest.raises(ValueError, match=r"q_1 = 1\.5"):
        LifeTable([0.1, 1.5, 1.0])
    with pytest.raises(ValueError, match="q_0"):
        LifeTable([-0.1, 1.0])
    with pytest.raises(ValueError):
        LifeTable([float("nan"), 1.0])
    with pytest.raises(ValueError):
        LifeTable([float("inf"), 1.0])


def test_table_must_close_and_only_at_the_end():
    with pytest.raises(ValueError, match="does not close"):
        LifeTable([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="closes the table at age"):
        LifeTable([0.1, 1.0, 0.3, 1.0])
    with pytest.raises(ValueError, match="empty"):
        LifeTable([])


def test_constructor_rejects_bad_radix_and_start_age():
    with pytest.raises(ValueError, match="radix"):
        LifeTable([0.5, 1.0], radix=0.0)
    with pytest.raises(ValueError, match="radix"):
        LifeTable([0.5, 1.0], radix=float("nan"))
    with pytest.raises(ValueError, match="start_age"):
        LifeTable([0.5, 1.0], start_age=-1)
    with pytest.raises(ValueError, match="start_age"):
        LifeTable([0.5, 1.0], start_age=20.5)


def test_from_lx_round_trips_and_keeps_survivors_exactly():
    ages, survivors = SULT.survival_curve()
    rebuilt = LifeTable.from_lx(survivors, start_age=ages[0])
    assert rebuilt.omega == SULT.omega
    assert rebuilt.radix == SULT.radix
    for x in SULT.ages:
        assert rebuilt.lx(x) == SULT.lx(x)          # stored verbatim
        assert rebuilt.qx(x) == pytest.approx(SULT.qx(x), rel=1e-9)


def test_from_lx_trims_trailing_zeros_and_validates():
    table = LifeTable.from_lx([100.0, 60.0, 20.0, 0.0, 0.0, 0.0], start_age=50)
    assert table.omega == 53
    assert table.ages == [50, 51, 52]
    with pytest.raises(ValueError, match="does not close"):
        LifeTable.from_lx([100.0, 60.0, 20.0])
    with pytest.raises(ValueError, match="non-increasing"):
        LifeTable.from_lx([100.0, 120.0, 0.0])
    with pytest.raises(ValueError, match="negative"):
        LifeTable.from_lx([100.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="radix"):
        LifeTable.from_lx([0.0, 0.0])
    with pytest.raises(ValueError, match="at least two"):
        LifeTable.from_lx([100.0])


def test_ages_outside_the_table_raise_with_the_range_in_the_message():
    with pytest.raises(ValueError, match="from 20 to 130"):
        SULT.qx(19)
    with pytest.raises(ValueError, match="omega = 131"):
        SULT.qx(131)
    with pytest.raises(ValueError, match="from 20 to 130"):
        SULT.lx(132)
    assert SULT.lx(131) == 0.0                     # l_omega is allowed, and zero
    with pytest.raises(ValueError, match="omega = 131"):
        SULT.dx(131)
    with pytest.raises(ValueError, match="whole number"):
        SULT.qx(40.5)
    with pytest.raises(ValueError, match="whole number"):
        SULT.tpx(2.5, 40)
    with pytest.raises(ValueError, match="t must be >= 0"):
        SULT.tpx(-1, 40)


# ---------------------------------------------------------------------------
# Survival probabilities and expectations
# ---------------------------------------------------------------------------


def test_tpx_is_multiplicative_and_bounded():
    assert SULT.tpx(0, 40) == 1.0
    assert SULT.tpx(1, 40) == pytest.approx(SULT.px(40), rel=1e-12)
    for s, t in ((5, 10), (20, 30), (0, 91)):
        assert SULT.tpx(s + t, 40) == pytest.approx(SULT.tpx(s, 40) * SULT.tpx(t, 40 + s), rel=1e-12)
    assert SULT.tpx(91, 40) == 0.0                 # age 131 = omega
    assert SULT.tpx(500, 40) == 0.0                # beyond the table is 0, not an error
    assert SULT.tqx(10, 40) == pytest.approx(1.0 - SULT.tpx(10, 40), abs=1e-15)


def test_deferred_mortality_is_a_distribution():
    total = sum(SULT.deferred_qx(t, 40) for t in range(0, SULT.omega - 40))
    assert total == pytest.approx(1.0, rel=1e-12)
    assert SULT.deferred_qx(0, 40) == pytest.approx(SULT.qx(40), rel=1e-12)
    assert SULT.deferred_qx(3, 40) == pytest.approx(SULT.tpx(3, 40) * SULT.qx(43), rel=1e-12)
    assert SULT.deferred_qx(91, 40) == 0.0


def test_curtate_expectation_is_the_sum_of_survival_probabilities():
    direct = sum(SULT.tpx(k, 40) for k in range(1, SULT.omega - 40))
    assert SULT.curtate_expectation(40) == pytest.approx(direct, rel=1e-12)
    assert SULT.complete_expectation(40) == pytest.approx(SULT.curtate_expectation(40) + 0.5)
    assert SULT.curtate_expectation(130) == 0.0    # the last age never completes another year
    # e_x = p_x (1 + e_{x+1}) — the recursion used to build expectation columns.
    assert SULT.curtate_expectation(60) == pytest.approx(
        SULT.px(60) * (1.0 + SULT.curtate_expectation(61)), rel=1e-12
    )


def test_curves_are_the_right_shape_for_plotting():
    ages, survivors = SULT.survival_curve()
    assert ages[0] == 20 and ages[-1] == SULT.omega
    assert len(ages) == len(survivors) == len(SULT) + 1
    assert survivors == sorted(survivors, reverse=True)
    assert survivors[-1] == 0.0
    ages_q, rates = SULT.mortality_curve()
    assert ages_q == SULT.ages
    assert len(rates) == len(SULT)
    assert rates[-1] == 1.0
    assert rates[:-1] == sorted(rates[:-1])        # Makeham mortality rises with age


# ---------------------------------------------------------------------------
# Parametric laws
# ---------------------------------------------------------------------------


def test_gompertz_is_makeham_with_zero_accident_hazard():
    g = LifeTable.gompertz(B=2.7e-6, c=1.124, start_age=20, max_age=110)
    m = LifeTable.makeham(A=0.0, B=2.7e-6, c=1.124, start_age=20, max_age=110)
    assert g.ages == m.ages
    for x in g.ages:
        assert g.qx(x) == m.qx(x)


def test_makeham_closed_form_matches_the_integrated_force():
    A, B, c = 0.00022, 2.7e-6, 1.124
    closed = LifeTable.makeham(A, B, c, start_age=20, max_age=110)
    numeric = LifeTable.from_force(lambda s: A + B * c ** s, start_age=20, max_age=110)
    assert closed.omega == numeric.omega == 111
    for x in (20, 45, 70, 95, 109):
        assert numeric.qx(x) == pytest.approx(closed.qx(x), abs=1e-11)
    for t, x in ((1, 20), (30, 40), (50, 50), (40, 70)):
        assert numeric.tpx(t, x) == pytest.approx(closed.tpx(t, x), abs=1e-10)


def test_from_force_matches_scipy_quadrature():
    integrate = pytest.importorskip("scipy.integrate")
    mu = lambda s: 0.0005 + 3e-5 * 1.1 ** s  # noqa: E731
    table = LifeTable.from_force(mu, start_age=30, max_age=100)
    for x, t in ((30, 10), (50, 25), (70, 20)):
        hazard, _ = integrate.quad(mu, x, x + t)
        assert table.tpx(t, x) == pytest.approx(math.exp(-hazard), rel=1e-9)


def test_parametric_constructors_validate():
    with pytest.raises(ValueError, match="B must be > 0"):
        LifeTable.gompertz(B=0.0, c=1.1)
    with pytest.raises(ValueError, match="c must be > 1"):
        LifeTable.gompertz(B=1e-5, c=1.0)
    with pytest.raises(ValueError, match="A must be >= 0"):
        LifeTable.makeham(A=-0.1, B=1e-5, c=1.1)
    with pytest.raises(ValueError, match="max_age must exceed start_age"):
        LifeTable.makeham(A=0.0, B=1e-5, c=1.1, start_age=50, max_age=50)
    with pytest.raises(ValueError, match="even"):
        LifeTable.from_force(lambda s: 0.01, steps=5)
    with pytest.raises(ValueError, match="steps must be >= 2"):
        LifeTable.from_force(lambda s: 0.01, steps=0)
    with pytest.raises(ValueError, match="negative"):
        LifeTable.from_force(lambda s: -0.01, max_age=30)
    with pytest.raises(ValueError, match="finite"):
        LifeTable.from_force(lambda s: float("inf"), max_age=30)
    with pytest.raises(TypeError, match="callable"):
        LifeTable.from_force(0.01)


def test_constant_force_from_force_is_exact():
    # Simpson is exact for a constant, so q_x = 1 - e^{-mu} to full precision.
    table = LifeTable.from_force(lambda s: 0.02, start_age=0, max_age=50, steps=2)
    assert table.qx(10) == pytest.approx(1.0 - math.exp(-0.02), rel=1e-14)
    assert table.tpx(30, 5) == pytest.approx(math.exp(-0.6), rel=1e-12)


def test_a_steep_law_closes_the_table_early_with_a_warning():
    with pytest.warns(RuntimeWarning, match="certainty"):
        table = LifeTable.gompertz(B=1e-4, c=1.15, start_age=0, max_age=130)
    assert table.omega < 131
    assert table.lx(table.omega) == 0.0
    assert table.qx(table.omega - 1) == 1.0
    assert sum(table.dx(x) for x in table.ages) == pytest.approx(table.radix, rel=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LifeTable.sult()                            # the standard table does not trip it


# ---------------------------------------------------------------------------
# SULT — published values and pinned values
# ---------------------------------------------------------------------------


def test_sult_reproduces_amlcr_appendix_d():
    # Dickson, Hardy & Waters, AMLCR 2nd ed., Tables D.1 and D.3 (i = 5%).
    assert SULT.lx(21) == pytest.approx(99975.04, abs=0.005)
    assert SULT.lx(50) == pytest.approx(98576.37, abs=0.005)
    assert SULT.lx(80) == pytest.approx(75657.16, abs=0.005)
    assert SULT.lx(100) == pytest.approx(6248.17, abs=0.005)
    assert life.whole_life_annuity_due(SULT, 20, I) == pytest.approx(19.9664, abs=5e-5)
    assert life.whole_life_annuity_due(SULT, 60, I) == pytest.approx(14.9041, abs=5e-5)
    assert life.whole_life_insurance(SULT, 20, I) == pytest.approx(0.04922, abs=5e-6)
    assert life.whole_life_insurance(SULT, 60, I) == pytest.approx(0.29028, abs=5e-6)
    assert life.pure_endowment(SULT, 20, 5, I) == pytest.approx(0.78252, abs=5e-6)


def test_sult_pinned_values():
    # Sanity ranges first, then the exact values this implementation gives.
    assert SULT.qx(20) == pytest.approx(0.000250, abs=5e-7)
    assert SULT.lx(100) < 0.07 * SULT.radix
    assert 55.0 < SULT.curtate_expectation(20) < 70.0
    assert SULT.qx(20) == pytest.approx(0.000249639028398585, rel=1e-6)
    assert SULT.lx(100) == pytest.approx(6248.174332519868, rel=1e-6)
    assert SULT.curtate_expectation(20) == pytest.approx(65.41315159665601, rel=1e-6)
    assert life.whole_life_annuity_due(SULT, 40, I) == pytest.approx(18.457756571743, rel=1e-6)
    assert life.whole_life_insurance(SULT, 40, I) == pytest.approx(0.12105921086937974, rel=1e-6)


# ---------------------------------------------------------------------------
# Interest
# ---------------------------------------------------------------------------


def test_interest_identities():
    v, d, delta = life.discount_factor(I), life.discount_rate(I), life.force_of_interest(I)
    assert v == pytest.approx(1.0 / 1.05, rel=1e-15)
    assert d == pytest.approx(I * v, rel=1e-15)
    assert d == pytest.approx(1.0 - v, rel=1e-15)
    assert delta == pytest.approx(math.log(1.05), rel=1e-15)
    assert math.exp(delta) == pytest.approx(1.0 + I, rel=1e-15)


@pytest.mark.parametrize("m", [1, 2, 4, 12, 365])
def test_nominal_and_effective_rates_round_trip(m):
    i_m = life.nominal_rate(I, m)
    assert life.effective_from_nominal(i_m, m) == pytest.approx(I, rel=1e-12)
    if m > 1:
        assert i_m < I                              # more frequent compounding, lower nominal rate
    else:
        assert i_m == pytest.approx(I)


def test_annuity_certain_against_the_formula_and_the_sum():
    n = 10
    immediate = life.annuity_certain(I, n)
    due = life.annuity_certain(I, n, due=True)
    assert immediate == pytest.approx((1.0 - V ** n) / I, rel=1e-14)
    assert due == pytest.approx((1.0 - V ** n) / D, rel=1e-14)
    assert immediate == pytest.approx(sum(V ** k for k in range(1, n + 1)), rel=1e-12)
    assert due == pytest.approx((1.0 + I) * immediate, rel=1e-14)
    assert life.annuity_certain(0.0, n) == n and life.annuity_certain(0.0, n, due=True) == n
    assert life.annuity_certain(I, 0) == 0.0


def test_accumulated_annuity_is_the_annuity_rolled_forward():
    n = 15
    assert life.accumulated_annuity(I, n) == pytest.approx(
        life.annuity_certain(I, n) * (1.0 + I) ** n, rel=1e-13
    )
    assert life.accumulated_annuity(I, n, due=True) == pytest.approx(
        life.annuity_certain(I, n, due=True) * (1.0 + I) ** n, rel=1e-13
    )
    assert life.accumulated_annuity(0.0, n) == n


def test_perpetuity_is_the_limit_of_a_long_annuity():
    assert life.perpetuity(I) == pytest.approx(life.annuity_certain(I, 5000), rel=1e-12)
    assert life.perpetuity(I, due=True) == pytest.approx(1.0 / D, rel=1e-15)
    with pytest.raises(ValueError, match="i > 0"):
        life.perpetuity(0.0)


def test_interest_validation():
    for bad in (-1.0, -1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="i must be"):
            life.discount_factor(bad)
    with pytest.raises(ValueError, match="m must be >= 1"):
        life.nominal_rate(I, 0)
    with pytest.raises(ValueError, match="exceed -1"):
        life.effective_from_nominal(-24.0, 12)
    with pytest.raises(ValueError, match="n must be >= 0"):
        life.annuity_certain(I, -1)
    with pytest.raises(ValueError, match="whole number"):
        life.annuity_certain(I, 2.5)
    # Negative but valid rates (deflation) are allowed everywhere.
    assert life.discount_factor(-0.01) > 1.0
    assert life.whole_life_annuity_due(SULT, 40, -0.01) > life.whole_life_annuity_due(SULT, 40, 0.0)


# ---------------------------------------------------------------------------
# Insurance and annuity identities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", [20, 40, 65, 90, 130])
def test_whole_life_insurance_equals_one_minus_d_times_annuity(x):
    A = life.whole_life_insurance(SULT, x, I)
    a = life.whole_life_annuity_due(SULT, x, I)
    assert A == pytest.approx(1.0 - D * a, abs=1e-12)


def test_values_at_the_last_age_are_certain():
    assert life.whole_life_insurance(SULT, 130, I) == pytest.approx(V, rel=1e-15)
    assert life.whole_life_annuity_due(SULT, 130, I) == 1.0
    assert life.whole_life_annuity_immediate(SULT, 130, I) == 0.0


def test_insurance_recursion():
    for x in (30, 55, 80, 110):
        lhs = life.whole_life_insurance(SULT, x, I)
        rhs = V * SULT.qx(x) + V * SULT.px(x) * life.whole_life_insurance(SULT, x + 1, I)
        assert lhs == pytest.approx(rhs, abs=1e-13)


def test_annuity_recursion():
    for x in (30, 55, 80, 110):
        lhs = life.whole_life_annuity_due(SULT, x, I)
        rhs = 1.0 + V * SULT.px(x) * life.whole_life_annuity_due(SULT, x + 1, I)
        assert lhs == pytest.approx(rhs, abs=1e-12)


def test_endowment_decomposes_into_term_plus_pure_endowment():
    x, n = 40, 25
    term = life.term_insurance(SULT, x, n, I)
    pure = life.pure_endowment(SULT, x, n, I)
    assert life.endowment_insurance(SULT, x, n, I) == pytest.approx(term + pure, abs=1e-13)
    assert pure == pytest.approx(V ** n * SULT.tpx(n, x), rel=1e-13)
    # The endowment identity holds for the continuous flag too, term part only accelerated.
    factor = I / math.log1p(I)
    assert life.endowment_insurance(SULT, x, n, I, continuous=True) == pytest.approx(
        factor * term + pure, abs=1e-13
    )


def test_term_and_deferred_split_whole_life():
    x, n = 50, 15
    A = life.whole_life_insurance(SULT, x, I)
    term = life.term_insurance(SULT, x, n, I)
    deferred = life.deferred_whole_life_insurance(SULT, x, n, I)
    assert term + deferred == pytest.approx(A, abs=1e-13)
    assert deferred == pytest.approx(
        life.pure_endowment(SULT, x, n, I) * life.whole_life_insurance(SULT, x + n, I), rel=1e-12
    )
    assert life.term_insurance(SULT, x, 0, I) == 0.0
    assert life.pure_endowment(SULT, x, 0, I) == 1.0
    assert life.term_insurance(SULT, x, 500, I) == pytest.approx(A, rel=1e-15)  # beyond omega = whole life
    assert life.deferred_whole_life_insurance(SULT, x, 500, I) == 0.0


def test_temporary_annuity_identities():
    x, n = 40, 20
    a_x = life.whole_life_annuity_due(SULT, x, I)
    a_xn = life.temporary_annuity_due(SULT, x, n, I)
    nE = life.pure_endowment(SULT, x, n, I)
    assert a_xn == pytest.approx(a_x - nE * life.whole_life_annuity_due(SULT, x + n, I), abs=1e-12)
    assert life.deferred_annuity_due(SULT, x, n, I) == pytest.approx(a_x - a_xn, abs=1e-12)
    assert life.temporary_annuity_immediate(SULT, x, n, I) == pytest.approx(a_xn - 1.0 + nE, abs=1e-13)
    assert life.temporary_annuity_due(SULT, x, 0, I) == 0.0
    assert life.temporary_annuity_due(SULT, x, 500, I) == pytest.approx(a_x, rel=1e-15)
    assert life.deferred_annuity_due(SULT, x, 500, I) == 0.0
    # Temporary annuity-due and endowment insurance obey the same 1 − d·ä relation.
    assert life.endowment_insurance(SULT, x, n, I) == pytest.approx(1.0 - D * a_xn, abs=1e-12)


def test_immediate_annuity_is_one_less_than_due():
    for x in (25, 60, 95):
        assert life.whole_life_annuity_immediate(SULT, x, I) == pytest.approx(
            life.whole_life_annuity_due(SULT, x, I) - 1.0, abs=1e-13
        )


def test_increasing_benefits_satisfy_their_recursions():
    for x in (30, 60, 100):
        IA = life.increasing_insurance(SULT, x, I)
        assert IA == pytest.approx(
            life.whole_life_insurance(SULT, x, I) + V * SULT.px(x) * life.increasing_insurance(SULT, x + 1, I),
            abs=1e-11,
        )
        Ia = life.increasing_annuity_due(SULT, x, I)
        assert Ia == pytest.approx(
            life.whole_life_annuity_due(SULT, x, I) + V * SULT.px(x) * life.increasing_annuity_due(SULT, x + 1, I),
            abs=1e-10,
        )
    assert life.increasing_insurance(SULT, 130, I) == pytest.approx(V, rel=1e-15)
    assert life.increasing_annuity_due(SULT, 130, I) == 1.0


def test_continuous_adjustments():
    x = 45
    factor = I / math.log1p(I)
    assert factor > 1.0
    assert life.whole_life_insurance(SULT, x, I, continuous=True) == pytest.approx(
        factor * life.whole_life_insurance(SULT, x, I), rel=1e-14
    )
    assert life.term_insurance(SULT, x, 10, I, continuous=True) == pytest.approx(
        factor * life.term_insurance(SULT, x, 10, I), rel=1e-14
    )
    assert life.deferred_whole_life_insurance(SULT, x, 10, I, continuous=True) == pytest.approx(
        factor * life.deferred_whole_life_insurance(SULT, x, 10, I), rel=1e-14
    )
    assert life.increasing_insurance(SULT, x, I, continuous=True) == pytest.approx(
        factor * life.increasing_insurance(SULT, x, I), rel=1e-14
    )
    # At zero interest there is nothing to accelerate.
    assert life.whole_life_insurance(SULT, x, 0.0, continuous=True) == pytest.approx(
        life.whole_life_insurance(SULT, x, 0.0), rel=1e-15
    )
    # Woolhouse two terms: the continuous annuity sits half a payment below the due one.
    a = life.whole_life_annuity_due(SULT, x, I)
    assert life.whole_life_annuity_due(SULT, x, I, continuous=True) == pytest.approx(a - 0.5)
    assert life.whole_life_annuity_immediate(SULT, x, I, continuous=True) == pytest.approx(a - 0.5)
    nE = life.pure_endowment(SULT, x, 20, I)
    a_n = life.temporary_annuity_due(SULT, x, 20, I)
    assert life.temporary_annuity_due(SULT, x, 20, I, continuous=True) == pytest.approx(a_n - 0.5 * (1 - nE))
    assert life.temporary_annuity_immediate(SULT, x, 20, I, continuous=True) == pytest.approx(a_n - 0.5 * (1 - nE))
    assert life.deferred_annuity_due(SULT, x, 20, I, continuous=True) == pytest.approx(
        life.deferred_annuity_due(SULT, x, 20, I) - 0.5 * nE
    )
    # The UDD-consistent continuous annuity (1 − Ā)/δ agrees with Woolhouse to the third decimal.
    udd = (1.0 - life.whole_life_insurance(SULT, x, I, continuous=True)) / math.log1p(I)
    assert life.whole_life_annuity_due(SULT, x, I, continuous=True) == pytest.approx(udd, abs=5e-3)


def test_mthly_annuity_is_woolhouse():
    x = 50
    a = life.whole_life_annuity_due(SULT, x, I)
    assert life.mthly_annuity_due(SULT, x, I, 1) == a
    assert life.mthly_annuity_due(SULT, x, I, 12) == pytest.approx(a - 11.0 / 24.0)
    assert life.mthly_annuity_due(SULT, x, I, 12) > life.whole_life_annuity_due(SULT, x, I, continuous=True)
    nE = life.pure_endowment(SULT, x, 10, I)
    assert life.mthly_annuity_due(SULT, x, I, 4, n=10) == pytest.approx(
        life.temporary_annuity_due(SULT, x, 10, I) - 3.0 / 8.0 * (1.0 - nE)
    )
    with pytest.raises(ValueError, match="m must be >= 1"):
        life.mthly_annuity_due(SULT, x, I, 0)


def test_apv_functions_validate_inputs():
    with pytest.raises(ValueError, match="outside the table"):
        life.whole_life_insurance(SULT, 10, I)
    with pytest.raises(ValueError, match="i must be"):
        life.whole_life_insurance(SULT, 40, -1.0)
    with pytest.raises(ValueError, match="n must be >= 0"):
        life.term_insurance(SULT, 40, -5, I)
    with pytest.raises(ValueError, match="whole number"):
        life.pure_endowment(SULT, 40, 2.5, I)
    with pytest.raises(ValueError, match="m must be >= 0"):
        life.deferred_annuity_due(SULT, 40, -1, I)
    with pytest.raises(TypeError, match="LifeTable"):
        life.whole_life_insurance([0.1, 1.0], 0, I)


# ---------------------------------------------------------------------------
# Commutation functions
# ---------------------------------------------------------------------------


def test_commutation_ratios_reproduce_the_direct_values():
    comm = life.Commutation(SULT, I)
    for x in (20, 45, 70, 100):
        assert comm.N(x) / comm.D(x) == pytest.approx(life.whole_life_annuity_due(SULT, x, I), rel=1e-12)
        assert comm.M(x) / comm.D(x) == pytest.approx(life.whole_life_insurance(SULT, x, I), rel=1e-12)
        assert comm.S(x) / comm.D(x) == pytest.approx(life.increasing_annuity_due(SULT, x, I), rel=1e-12)
        assert comm.R(x) / comm.D(x) == pytest.approx(life.increasing_insurance(SULT, x, I), rel=1e-12)
        n = 20
        assert comm.D(x + n) / comm.D(x) == pytest.approx(life.pure_endowment(SULT, x, n, I), rel=1e-12)
        assert (comm.M(x) - comm.M(x + n)) / comm.D(x) == pytest.approx(life.term_insurance(SULT, x, n, I), rel=1e-11)
        assert (comm.N(x) - comm.N(x + n)) / comm.D(x) == pytest.approx(life.temporary_annuity_due(SULT, x, n, I), rel=1e-12)
    assert comm.D(20) == pytest.approx(V ** 20 * SULT.radix, rel=1e-15)
    assert comm.C(40) == pytest.approx(V ** 41 * SULT.dx(40), rel=1e-13)
    assert comm.D(131) == comm.N(131) == comm.C(131) == comm.M(131) == comm.S(131) == comm.R(131) == 0.0
    with pytest.raises(ValueError, match="outside the table"):
        comm.D(132)
    assert "i=0.05" in repr(comm)


# ---------------------------------------------------------------------------
# Premiums
# ---------------------------------------------------------------------------


def test_net_premiums_are_benefit_over_annuity():
    x, n = 40, 20
    assert life.whole_life_premium(SULT, x, I) == pytest.approx(
        life.whole_life_insurance(SULT, x, I) / life.whole_life_annuity_due(SULT, x, I), rel=1e-14
    )
    assert life.term_premium(SULT, x, n, I) == pytest.approx(
        life.term_insurance(SULT, x, n, I) / life.temporary_annuity_due(SULT, x, n, I), rel=1e-14
    )
    assert life.endowment_premium(SULT, x, n, I) == pytest.approx(
        life.endowment_insurance(SULT, x, n, I) / life.temporary_annuity_due(SULT, x, n, I), rel=1e-14
    )
    assert life.term_premium(SULT, x, n, I) < life.whole_life_premium(SULT, x, I) < life.endowment_premium(SULT, x, n, I)
    # P_x = 1/ä_x − d is the exam shortcut, and follows from A_x = 1 − d·ä_x.
    assert life.whole_life_premium(SULT, x, I) == pytest.approx(
        1.0 / life.whole_life_annuity_due(SULT, x, I) - D, abs=1e-14
    )
    assert life.net_premium(2.0, 4.0) == 0.5
    with pytest.raises(ValueError, match="annuity_apv must be > 0"):
        life.net_premium(1.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        life.net_premium(float("nan"), 1.0)
    with pytest.raises(ValueError, match="n must be >= 1"):
        life.term_premium(SULT, x, 0, I)


def test_gross_premium_reduces_to_net_and_loads_correctly():
    x = 40
    P = life.whole_life_premium(SULT, x, I)
    A = life.whole_life_insurance(SULT, x, I)
    a = life.whole_life_annuity_due(SULT, x, I)
    assert life.gross_premium(SULT, x, I, benefit=100_000) == pytest.approx(100_000 * P, rel=1e-13)
    G = life.gross_premium(
        SULT, x, I, benefit=100_000, initial_expense=500.0, renewal_expense=50.0,
        premium_loading=0.03, initial_loading=0.5, settlement_expense=200.0,
    )
    # Reconstruct the equivalence: income equals outgo.
    income = G * a * (1.0 - 0.03) - 0.5 * G
    outgo = (100_000 + 200.0) * A + 500.0 + 50.0 * (a - 1.0)
    assert income == pytest.approx(outgo, rel=1e-12)
    base = life.gross_premium(SULT, x, I, benefit=100_000)
    for kwargs in (
        dict(initial_expense=100.0), dict(renewal_expense=10.0), dict(premium_loading=0.02),
        dict(initial_loading=0.3), dict(settlement_expense=100.0),
    ):
        assert life.gross_premium(SULT, x, I, benefit=100_000, **kwargs) > base
    with pytest.raises(ValueError, match="premium_loading"):
        life.gross_premium(SULT, x, I, premium_loading=1.0)
    with pytest.raises(ValueError, match="initial_expense must be >= 0"):
        life.gross_premium(SULT, x, I, initial_expense=-1.0)
    with pytest.raises(ValueError, match="benefit must be > 0"):
        life.gross_premium(SULT, x, I, benefit=0.0)
    with pytest.raises(ValueError, match="swallow"):
        life.gross_premium(SULT, 130, I, premium_loading=0.5, initial_loading=0.6)


# ---------------------------------------------------------------------------
# Reserves
# ---------------------------------------------------------------------------


def test_reserves_at_issue_and_at_expiry():
    x, n = 40, 20
    assert life.net_premium_reserve(SULT, x, 0, I) == pytest.approx(0.0, abs=1e-12)
    assert life.net_premium_reserve(SULT, x, 0, I, "term", n) == pytest.approx(0.0, abs=1e-12)
    assert life.net_premium_reserve(SULT, x, 0, I, "endowment", n) == pytest.approx(0.0, abs=1e-12)
    assert life.net_premium_reserve(SULT, x, n, I, "endowment", n) == 1.0
    assert life.net_premium_reserve(SULT, x, n, I, "term", n) == 0.0
    # The last whole-life reserve: one certain payment of v, less the premium just received.
    last = SULT.omega - x - 1
    assert life.net_premium_reserve(SULT, x, last, I) == pytest.approx(V - life.whole_life_premium(SULT, x, I), rel=1e-12)


@pytest.mark.parametrize("kind,n", [("whole_life", None), ("term", 20), ("endowment", 20)])
def test_fackler_recursion(kind, n):
    x = 40
    P = life._premium_for(SULT, x, I, kind, n)
    last = (SULT.omega - x - 1) if kind == "whole_life" else n
    for t in range(last):
        now = life.net_premium_reserve(SULT, x, t, I, kind, n)
        nxt = life.net_premium_reserve(SULT, x, t + 1, I, kind, n)
        lhs = (now + P) * (1.0 + I)
        rhs = SULT.qx(x + t) + SULT.px(x + t) * nxt
        assert lhs == pytest.approx(rhs, abs=1e-11), (kind, t)


def test_reserve_profile_matches_pointwise_values_and_builds_up():
    x, n = 40, 20
    profile = life.reserve_profile(SULT, x, I, "endowment", n)
    assert [t for t, _ in profile] == list(range(n + 1))
    for t, value in profile:
        assert value == pytest.approx(life.net_premium_reserve(SULT, x, t, I, "endowment", n), rel=1e-12)
    values = [v for _, v in profile]
    assert values == sorted(values)                 # an endowment reserve rises monotonically to 1
    whole = life.reserve_profile(SULT, x, I)
    assert whole[0] == (0, pytest.approx(0.0, abs=1e-12))
    assert whole[-1][0] == SULT.omega - x - 1
    assert 0.0 < whole[30][1] < whole[60][1] < 1.0
    term = life.reserve_profile(SULT, x, I, "term", n)
    assert term[-1] == (n, 0.0)
    assert max(v for _, v in term) < 0.05           # term reserves are small


def test_reserve_validation():
    with pytest.raises(ValueError, match="kind must be one of"):
        life.net_premium_reserve(SULT, 40, 5, I, "universal")
    with pytest.raises(ValueError, match="needs a term"):
        life.net_premium_reserve(SULT, 40, 5, I, "term")
    with pytest.raises(ValueError, match="whole_life has no term"):
        life.net_premium_reserve(SULT, 40, 5, I, "whole_life", 20)
    with pytest.raises(ValueError, match="expired"):
        life.net_premium_reserve(SULT, 40, 21, I, "endowment", 20)
    with pytest.raises(ValueError, match="t must be >= 0"):
        life.net_premium_reserve(SULT, 40, -1, I)
    with pytest.raises(ValueError, match="past the table"):
        life.net_premium_reserve(SULT, 40, 91, I)
    with pytest.raises(ValueError, match="runs past omega"):
        life.net_premium_reserve(SULT, 100, 5, I, "endowment", 40)
    with pytest.raises(ValueError, match="n must be >= 1"):
        life.reserve_profile(SULT, 40, I, "term", 0)
    with pytest.raises(ValueError, match="outside the table"):
        life.reserve_profile(SULT, 10, I)


# ---------------------------------------------------------------------------
# Joint life
# ---------------------------------------------------------------------------


def test_joint_and_last_survivor_annuities_sum_to_the_singles():
    x, y = 65, 60
    joint = life.joint_life_annuity_due(SULT, x, y, I)
    last = life.last_survivor_annuity_due(SULT, x, y, I)
    a_x = life.whole_life_annuity_due(SULT, x, I)
    a_y = life.whole_life_annuity_due(SULT, y, I)
    assert joint + last == pytest.approx(a_x + a_y, abs=1e-12)
    assert joint < min(a_x, a_y) < max(a_x, a_y) < last
    assert joint == pytest.approx(life.joint_life_annuity_due(SULT, y, x, I), rel=1e-14)   # symmetric
    assert joint == pytest.approx(
        sum(V ** k * SULT.tpx(k, x) * SULT.tpx(k, y) for k in range(0, SULT.omega - x)), rel=1e-12
    )


def test_joint_life_insurance_identity():
    x, y = 65, 60
    A_xy = life.joint_life_insurance(SULT, x, y, I)
    assert A_xy == pytest.approx(1.0 - D * life.joint_life_annuity_due(SULT, x, y, I), abs=1e-12)
    A_last = life.last_survivor_insurance(SULT, x, y, I)
    assert A_xy + A_last == pytest.approx(
        life.whole_life_insurance(SULT, x, I) + life.whole_life_insurance(SULT, y, I), abs=1e-12
    )
    assert A_last < life.whole_life_insurance(SULT, x, I) < A_xy
    # A life paired with itself at the last age dies for certain this year.
    assert life.joint_life_insurance(SULT, 130, 60, I) == pytest.approx(V, rel=1e-14)
    assert life.joint_life_annuity_due(SULT, 130, 60, I) == 1.0


def test_joint_life_with_a_second_table():
    lighter = LifeTable.makeham(0.0001, 2.0e-6, 1.124, start_age=20, max_age=130)
    joint_same = life.joint_life_annuity_due(SULT, 65, 65, I)
    joint_mixed = life.joint_life_annuity_due(SULT, 65, 65, I, table_y=lighter)
    assert joint_mixed > joint_same
    assert joint_mixed + life.last_survivor_annuity_due(SULT, 65, 65, I, table_y=lighter) == pytest.approx(
        life.whole_life_annuity_due(SULT, 65, I) + life.whole_life_annuity_due(lighter, 65, I), abs=1e-12
    )
    assert life.joint_life_insurance(SULT, 65, 65, I, table_y=lighter) == pytest.approx(
        1.0 - D * joint_mixed, abs=1e-12
    )
    with pytest.raises(ValueError, match="outside the table"):
        life.joint_life_annuity_due(SULT, 65, 10, I)
    with pytest.raises(TypeError, match="LifeTable"):
        life.joint_life_annuity_due(SULT, 65, 60, I, table_y="female")


# ---------------------------------------------------------------------------
# Verification hook
# ---------------------------------------------------------------------------


def test_verification_checks_pass():
    checks = life._verification_checks()
    assert 4 <= len(checks) <= 8
    for name, value, reference, tolerance in checks:
        assert isinstance(name, str) and name
        assert math.isfinite(value) and math.isfinite(reference)
        assert abs(value - reference) <= tolerance, name


def test_all_exports_exist():
    for name in life.__all__:
        assert hasattr(life, name), name
    assert not any(name.startswith("_") for name in life.__all__)
