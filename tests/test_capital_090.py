"""0.9.0 fixes to the copulas in riskpy.capital, from the machine-testing findings on 0.8.0.

- #7: ``joint_exceedance`` without the cancellation of 1 − 2q + C(q, q), for every family.
- #8: ``FrankCopula.cdf`` at a tiny |theta| (it collapsed below about 1e-160).
- #9: ``GaussianCopula.cdf`` within the Fréchet bounds and monotone at strong negative correlation.

Reference values were computed with mpmath at 50 digits, at the exact floating-point inputs the
library sees (for a level q that is 1 − q, which for q = 1 − 1e-8 is 1.000000005e-8, not 1e-8):
the elliptical ones by the Plackett integral from the countermonotone end, checked against the
conditional-distribution integral and, for the t at rho = 0, the chi-square mixing integral; the
Archimedean ones by the closed forms with enough digits to absorb the cancellation.
"""
import math
import time

import pytest

from riskpy import capital, tools
from riskpy.capital import ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula, StudentTCopula


def _rel(got, exact):
    return abs(got - exact) / abs(exact)


# ---------------------------------------------------------------------------
# #7 joint exceedance


@pytest.mark.parametrize("cop", [GumbelCopula(1.0), GumbelCopula(1.0, survival=True), GaussianCopula(0.0)], ids=repr)
@pytest.mark.parametrize("q", [0.9, 0.995, 1 - 1e-6, 1 - 1e-8, 1 - 1e-9, 1 - 1e-12])
def test_independence_copulas_give_the_square_of_the_tail(cop, q):
    assert _rel(cop.joint_exceedance(q), (1.0 - q) ** 2) < 1e-12


@pytest.mark.parametrize("cop", [StudentTCopula(0.0, df=1e6), FrankCopula(1e-6), FrankCopula(-1e-6), ClaytonCopula(1e-6),
                                 ClaytonCopula(1e-6, survival=True)], ids=repr)
@pytest.mark.parametrize("q", [1 - 1e-8, 1 - 1e-9])
def test_near_independence_at_high_levels(cop, q):
    # 0.8.0 returned one or two ulps of 1 (1.1e-16 or 2.2e-16) at 1 - 1e-8 and 0 at 1 - 1e-9
    assert cop.joint_exceedance(q) == pytest.approx((1.0 - q) ** 2, rel=1e-3)


_JE_REFERENCE = [
    # negative correlation at practical levels: 0.8.0 gave 7.012e-13 and 5.55e-16 (18x) for the Gaussian,
    # and 0 for the t at df = 1e300
    (GaussianCopula(-0.7), 0.995, 7.0232562325533554343e-13, 1e-12),
    (GaussianCopula(-0.7), 0.999, 3.0019405446216340821e-17, 1e-12),
    (StudentTCopula(-0.7, df=1e300), 0.995, 7.0232562325533554343e-13, 1e-12),
    (StudentTCopula(-0.7, df=1e300), 0.999, 3.0019405446216340821e-17, 1e-12),
    (StudentTCopula(-0.7, df=4.0), 0.995, 2.0092490924000302904e-05, 1e-12),
    (StudentTCopula(0.0, df=1e6), 1 - 1e-8, 1.0005262864427575103e-16, 1e-12),
    (StudentTCopula(0.0, df=1e6), 1 - 1e-9, 1.0006818631926105385e-18, 1e-12),
    (FrankCopula(-5.0), 1 - 1e-6, 3.3918444125336279174e-14, 1e-13),
    (FrankCopula(5.0), 1 - 1e-6, 5.0338931053664150071e-12, 1e-13),
    (FrankCopula(-500.0), 0.9, 3.8303391934279263394e-177, 1e-13),
    (FrankCopula(1e-6), 1 - 1e-8, 1.000000510049596934e-16, 1e-13),
    (ClaytonCopula(2.0), 1 - 1e-7, 2.9999993968419908615e-14, 1e-13),
    (ClaytonCopula(1e-6), 1 - 1e-9, 1.0000009434360803092e-18, 1e-13),
    (ClaytonCopula(2.0, survival=True), 1 - 1e-7, 7.0710678081435950287e-8, 1e-13),
    (GumbelCopula(1.5, survival=True), 1 - 1e-7, 7.730233916727886807e-12, 1e-12),
]


@pytest.mark.parametrize("cop, q, exact, tol", _JE_REFERENCE, ids=lambda x: repr(x) if not isinstance(x, float) else None)
def test_joint_exceedance_matches_mpmath_relatively(cop, q, exact, tol):
    assert _rel(cop.joint_exceedance(q), exact) < tol


def test_documented_gumbel_value_is_unchanged():
    assert GumbelCopula(2.0).joint_exceedance(0.995) == pytest.approx(0.002936262, rel=1e-6)


_FAMILIES = [GaussianCopula(-0.999), GaussianCopula(-0.3), GaussianCopula(0.6), StudentTCopula(-0.9, df=2.0),
             StudentTCopula(0.5, df=0.5), StudentTCopula(0.95, df=40.0), ClaytonCopula(0.3), ClaytonCopula(80.0),
             ClaytonCopula(5.0, survival=True), GumbelCopula(1.0 + 1e-9), GumbelCopula(30.0),
             GumbelCopula(3.0, survival=True), FrankCopula(-500.0), FrankCopula(-0.4), FrankCopula(0.4),
             FrankCopula(450.0)]


@pytest.mark.parametrize("cop", _FAMILIES, ids=repr)
def test_joint_exceedance_stays_within_the_frechet_bounds_and_falls_with_the_level(cop):
    levels = [0.01, 0.2, 0.5, 0.8, 0.95, 0.995, 1 - 1e-6, 1 - 1e-10]
    values = [cop.joint_exceedance(q) for q in levels]
    for q, je in zip(levels, values):
        assert max(1.0 - 2.0 * q, 0.0) <= je <= 1.0 - q
    assert all(b <= a for a, b in zip(values, values[1:]))


def test_copula_dependence_reports_a_near_independent_frank_as_independent():
    out = tools.call("copula_dependence", {"family": "frank", "copula_theta": 1e-6, "level": 1 - 1e-8})
    assert out["multiple_of_independent"] == pytest.approx(1.0, abs=1e-5)       # 0.8.0: 2.22


# ---------------------------------------------------------------------------
# #8 Frank at a tiny theta


@pytest.mark.parametrize("theta", [1e-200, -1e-200, 1e-300, -1e-300, 5e-324, -5e-324, 1e-160, -1e-12])
def test_frank_with_a_tiny_theta_is_the_independence_copula(theta):
    cop = FrankCopula(theta)
    assert cop.cdf(0.5, 0.5) == pytest.approx(0.25, rel=1e-12)
    assert cop.cdf(0.3, 0.9) == pytest.approx(0.27, rel=1e-11)
    assert cop.cdf(1e-300, 1e-6) == pytest.approx(1e-306, rel=1e-12)
    assert cop.joint_exceedance(0.995) == pytest.approx(0.005 ** 2, rel=1e-12)


def test_frank_tiny_theta_through_the_tool():
    out = tools.call("copula_dependence", {"family": "frank", "copula_theta": 1e-200, "level": 0.995})
    assert out["joint_exceedance"] == pytest.approx(2.5e-5, rel=1e-12)          # 0.8.0: 0.0


@pytest.mark.parametrize("theta", [1e-9, 2e-5, 0.5, 0.999999, 1.000001, 3.0, -1e-9, -2e-5, -0.5, -3.0])
@pytest.mark.parametrize("u, v", [(0.3, 0.9), (1e-6, 0.5), (0.999, 0.998), (0.5, 1e-9)])
def test_frank_cdf_matches_the_series_and_the_closed_form_across_its_switches(theta, u, v):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(60):
        t = mp.mpf(theta)
        exact = float(-1 / t * mp.log(1 + mp.expm1(-t * u) * mp.expm1(-t * v) / mp.expm1(-t)))
    assert _rel(FrankCopula(theta).cdf(u, v), exact) < 1e-14


def test_negative_frank_keeps_its_digits_in_the_corner():
    # 0.8.0 reflected the positive copula, u - C(u, 1 - v), which kept absolute precision only
    assert _rel(FrankCopula(-20.0).cdf(1e-6, 0.3), 8.2947586191487890501e-13) < 1e-13


# ---------------------------------------------------------------------------
# #9 Gaussian cdf: Fréchet bounds and monotonicity


@pytest.mark.parametrize("rho", [-0.9999, -0.999, -0.99, -0.9, -0.5])
def test_gaussian_cdf_respects_both_frechet_bounds_and_is_monotone(rho):
    cop = GaussianCopula(rho)
    grid = [1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 0.999999]
    for u in grid:
        row = [cop.cdf(u, v) for v in grid]
        for v, c in zip(grid, row):
            assert max(u + v - 1.0, 0.0) <= c <= min(u, v)
        assert all(b >= a for a, b in zip(row, row[1:]))


def test_gaussian_cdf_at_the_documented_violation():
    cop = GaussianCopula(-0.99)
    for u, v in ((0.5, 0.9), (0.3, 0.99), (0.9, 0.9), (0.7, 0.7)):
        assert cop.cdf(u, v) >= u + v - 1.0                   # 0.8.0: 0.3999999999999937 at (0.5, 0.9)


def test_gaussian_cdf_random_grid_is_2_increasing():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(90)
    for rho in (-0.995, -0.6, 0.0, 0.6, 0.995):
        cop = GaussianCopula(rho)
        for _ in range(25):
            u1, u2 = sorted(rng.uniform(0, 1, 2))
            v1, v2 = sorted(rng.uniform(0, 1, 2))
            c11, c12, c21, c22 = cop.cdf(u1, v1), cop.cdf(u1, v2), cop.cdf(u2, v1), cop.cdf(u2, v2)
            assert c21 >= c11 and c12 >= c11
            assert c22 - c12 - c21 + c11 >= -1e-15


@pytest.mark.parametrize("rho, u, v, exact", [
    # a + b is tiny but not zero, so the integrand near -pi/2 is not zero: cos(theta)^2, not (1 - sin)(1 + sin)
    (-0.99, 0.05, 0.95, 0.0058105136698882818414),
    (-0.999, 0.7, 0.05, 3.1050426026742995157e-142),       # a value whose integrand carries rounding noise
])
def test_gaussian_cdf_corner_values_match_mpmath(rho, u, v, exact):
    assert _rel(GaussianCopula(rho).cdf(u, v), exact) < 1e-11


# ---------------------------------------------------------------------------
# Survival copulas in their own corner, and the integrator's bound on work


@pytest.mark.parametrize("cop, u, v, exact", [
    (GumbelCopula(200.0, survival=True), 1e-12, 0.5, 9.9999999999999997989e-13),
    (GumbelCopula(1.0 + 1e-6, survival=True), 1e-6, 1e-3, 1.0079008101193488603e-9),
    (ClaytonCopula(3.0, survival=True), 1e-9, 0.2, 5.9039999960023045932e-10),
], ids=repr)
def test_survival_copulas_keep_their_digits_in_the_corner(cop, u, v, exact):
    assert _rel(cop.cdf(u, v), exact) < 1e-13


def test_extreme_elliptical_inputs_answer_promptly():
    # (A t quantile beyond 1e150, e.g. u = 1e-300 at df = 1, is refused with RP304 by design, so the
    # t copulas stop at 1e-12.)
    started = time.perf_counter()
    for cop, tiny in ((GaussianCopula(-0.999999), 1e-300), (GaussianCopula(0.999999), 1e-300),
                      (StudentTCopula(-0.999999, df=1.0), 1e-12), (StudentTCopula(0.999999, df=1e5), 1e-12),
                      (StudentTCopula(-0.5, df=0.3), 1e-12)):
        for u, v in ((tiny, 0.5), (0.5, 1 - 1e-12), (tiny, tiny), (0.3, 0.7)):
            c = cop.cdf(u, v)
            assert math.isfinite(c) and max(u + v - 1.0, 0.0) <= c <= min(u, v)
        assert 0.0 <= cop.joint_exceedance(1 - 1e-12) <= 1e-12
    assert time.perf_counter() - started < 20.0


def test_adaptive_simpson_has_a_work_budget():
    calls = [0]

    def noisy(x):                    # rounding-level noise everywhere: no tolerance can be met
        calls[0] += 1
        return 1.0 + 1e-9 * math.sin(1e9 * x)

    capital._adaptive_simpson(noisy, 0.0, 1.0, rel=1e-16, floor=0.0)
    assert calls[0] <= capital._SIMPSON_BUDGET + 200
