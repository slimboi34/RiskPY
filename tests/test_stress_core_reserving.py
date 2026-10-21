"""Regression and stress tests: compiled-core fixes (health-claims frequency, FFT
grid sizing, lognormal LER, exposure-rating domain, LossTriangle zero column, NaN
checks) plus the Python fixes in ``mc`` (Expected Shortfall on atoms, tiny-shape
quantiles) and ``reserving`` (φ = 0 bootstrap, numpy scalars, bool arguments,
non-numeric DataFrames).

The compiled-core tests need a core with these fixes (the release after 0.5.2).
Against an older wheel — the no-toolchain overlay recipe — they are skipped.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

import riskpy  # noqa: E402
from riskpy import (  # noqa: E402
    ActuarialMath,
    ExperienceRating,
    ExposureRating,
    FourierTransform,
    LossTriangle,
    MonteCarloSimulator,
    RateAnalyzer,
    mc,
    reserving,
)
from riskpy.errors import InputError  # noqa: E402


def _version_tuple(v):
    return tuple(int(p) for p in v.split("+")[0].split(".")[:3] if p.isdigit())


def _core_has_fixes() -> bool:
    # The published 0.5.2 core silently uses a factor of 1.0 here; the fixed one raises.
    tri = LossTriangle()
    tri.add_origin_year(0, [0.0, 1.0])
    tri.add_origin_year(1, [0.0])
    try:
        tri.get_development_factors()
    except ValueError:
        return True
    return False


needs_053_core = pytest.mark.skipif(
    not (_version_tuple(riskpy.__version__) >= (0, 5, 3) or _core_has_fixes()),
    reason="compiled core older than 0.5.3",
)

NAN = float("nan")
INF = float("inf")


# ---------------------------------------------------------------------------
# 1. simulate_health_claims: negative binomial with a fractional 1/dispersion
# ---------------------------------------------------------------------------


@needs_053_core
@pytest.mark.parametrize("dispersion", [0.3, 0.5, 1.7, 3.0])
def test_health_claims_negative_binomial_moments(dispersion):
    # Unit severity (gamma shape huge, scale tiny ≈ 1 each) would be slow; use
    # Gamma(1, 1) severity and check the aggregate moments instead:
    # E[S] = λ, Var[S] = E[N]·Var X + Var N·(E X)² = λ + (λ + dλ²).
    lam = 10.0
    x = np.asarray(MonteCarloSimulator(200_000, 11).simulate_health_claims(lam, dispersion, 1.0, 1.0))
    var_theory = lam + lam + dispersion * lam ** 2
    assert x.mean() == pytest.approx(lam, rel=0.01)
    assert x.var() == pytest.approx(var_theory, rel=0.04)


@needs_053_core
def test_health_claims_dispersion_above_one_is_not_all_zero():
    x = np.asarray(MonteCarloSimulator(50_000, 3).simulate_health_claims(10.0, 2.5, 2.0, 50.0))
    assert (x > 0).mean() > 0.3
    assert x.mean() == pytest.approx(10.0 * 100.0, rel=0.05)


@needs_053_core
def test_health_claims_tiny_dispersion_is_poisson():
    x = np.asarray(MonteCarloSimulator(100_000, 5).simulate_health_claims(10.0, 1e-300, 1.0, 1.0))
    assert x.mean() == pytest.approx(10.0, rel=0.01)
    assert x.var() == pytest.approx(20.0, rel=0.04)


@needs_053_core
@pytest.mark.parametrize("dispersion", [0.0, -1.0, NAN, INF])
def test_health_claims_rejects_bad_dispersion(dispersion):
    with pytest.raises(ValueError):
        MonteCarloSimulator(10, 1).simulate_health_claims(10.0, dispersion, 1.0, 1.0)


# ---------------------------------------------------------------------------
# 2. FFT compound Poisson: default grid sized from λ, wrap-around detected
# ---------------------------------------------------------------------------


def _poisson_pmf(lam, k):
    return np.exp(-lam + k * math.log(lam) - np.array([math.lgamma(i + 1.0) for i in k]))


@needs_053_core
@pytest.mark.parametrize("lam", [0.1, 0.5, 1.0, 3.0, 10.0, 25.0, 50.0, 100.0, 200.0])
def test_fft_default_grid_matches_poisson(lam):
    pmf = np.asarray(FourierTransform.compound_poisson_pmf([0.0, 1.0], lam))
    k = np.arange(pmf.size)
    assert (k * pmf).sum() == pytest.approx(lam, rel=1e-9)
    assert np.abs(pmf - _poisson_pmf(lam, k)).max() < 1e-12


@needs_053_core
def test_fft_default_grid_wide_severity():
    rng = np.random.default_rng(7)
    sev = rng.random(300)
    sev /= sev.sum()
    mean_sev = float((np.arange(sev.size) * sev).sum())
    for lam in [0.3, 5.0, 40.0]:
        pmf = np.asarray(FourierTransform.compound_poisson_pmf(sev.tolist(), lam))
        assert (np.arange(pmf.size) * pmf).sum() == pytest.approx(lam * mean_sev, rel=1e-8)


@needs_053_core
def test_fft_small_explicit_grid_that_wraps_raises():
    # Used to return a PMF with mean 7.45 instead of 50.
    with pytest.raises(ValueError, match="wrapped"):
        FourierTransform.compound_poisson_pmf([0.0, 1.0], 50.0, 16)


@needs_053_core
def test_fft_absurd_grid_rejected():
    with pytest.raises(ValueError, match="2\\^24"):
        FourierTransform.compound_poisson_pmf([0.0, 1.0], 1.0, 1 << 30)


# ---------------------------------------------------------------------------
# 3. ExposureRating.loss_elimination_ratio is the lognormal LER
# ---------------------------------------------------------------------------


def _lognormal_ler_reference(d, mean, cv):
    integrate = pytest.importorskip("scipy.integrate")
    stats = pytest.importorskip("scipy.stats")
    s2 = math.log1p(cv * cv)
    s = math.sqrt(s2)
    dist = stats.lognorm(s, scale=math.exp(math.log(mean) - s2 / 2.0))
    lev, _ = integrate.quad(dist.sf, 0.0, d, limit=200, epsabs=1e-12, epsrel=1e-12)
    return lev / mean


@needs_053_core
def test_ler_known_values():
    er = ExposureRating()
    for d, e, cv in [(500, 1000, 1.0), (2000, 1000, 2.0), (1000, 1000, 0.5), (100, 1000, 3.0)]:
        assert er.loss_elimination_ratio(d, e, cv) == pytest.approx(_lognormal_ler_reference(d, e, cv), rel=1e-8)


@needs_053_core
def test_ler_random_against_quadrature():
    rng = np.random.default_rng(2024)
    er = ExposureRating()
    for _ in range(40):
        e = float(10 ** rng.uniform(1, 6))
        cv = float(rng.uniform(0.1, 4.0))
        d = float(e * 10 ** rng.uniform(-2, 1))
        assert er.loss_elimination_ratio(d, e, cv) == pytest.approx(
            _lognormal_ler_reference(d, e, cv), rel=1e-6, abs=1e-10)


@needs_053_core
def test_ler_limits_and_validation():
    er = ExposureRating()
    assert er.loss_elimination_ratio(0.0, 1000.0, 1.0) == 0.0
    assert er.loss_elimination_ratio(1e12, 1000.0, 1.0) == pytest.approx(1.0, abs=1e-9)
    grid = [er.loss_elimination_ratio(d, 1000.0, 1.5) for d in np.geomspace(1, 1e6, 50)]
    assert all(b >= a for a, b in zip(grid, grid[1:]))
    for bad in [(NAN, 1000.0, 1.0), (-1.0, 1000.0, 1.0), (10.0, NAN, 1.0), (10.0, 1000.0, NAN), (10.0, 1000.0, INF)]:
        with pytest.raises(ValueError):
            er.loss_elimination_ratio(*bad)


# ---------------------------------------------------------------------------
# 7. ExposureRating domain: b in [0, 1], finite limits
# ---------------------------------------------------------------------------


@needs_053_core
@pytest.mark.parametrize("b", [-0.1, 1.5, NAN])
def test_exposure_rating_rejects_b_outside_unit_interval(b):
    with pytest.raises(ValueError):
        ExposureRating(1000.0, b)


@needs_053_core
def test_exposure_rating_rejects_non_finite_limits():
    er = ExposureRating(1000.0, 0.5)
    for bad in [NAN, INF]:
        with pytest.raises(ValueError):
            er.increased_limits_factor(bad)
        with pytest.raises(ValueError):
            er.layer_premium(100.0, 0.0, bad)
        with pytest.raises(ValueError):
            ExposureRating(bad, 0.5)


def test_layer_premiums_non_negative_across_valid_b():
    rng = np.random.default_rng(1)
    for b in np.linspace(0.0, 1.0, 11):
        er = ExposureRating(1000.0, float(b))
        for _ in range(20):
            att, lim = float(rng.uniform(0, 1e5)), float(rng.uniform(1, 1e5))
            assert er.layer_premium(100.0, att, lim) >= -1e-9


# ---------------------------------------------------------------------------
# 6. LossTriangle zero column; random triangles agree with Python
# ---------------------------------------------------------------------------


@needs_053_core
def test_loss_triangle_zero_then_positive_column_raises():
    tri = LossTriangle()
    tri.add_origin_year(0, [0.0, 10.0, 20.0])
    tri.add_origin_year(1, [0.0, 5.0])
    tri.add_origin_year(2, [7.0])
    with pytest.raises(ValueError, match="column 0"):
        tri.get_ultimate_losses()
    with pytest.raises(InputError):
        reserving.chain_ladder([[0, 10, 20], [0, 5], [7]])


def test_loss_triangle_all_zero_columns_still_factor_one():
    # Nothing paid at either age is no development, not an error.
    tri = LossTriangle()
    tri.add_origin_year(0, [0.0, 0.0, 0.0])
    tri.add_origin_year(1, [0.0, 0.0])
    tri.add_origin_year(2, [5.0])
    assert list(tri.get_development_factors()) == [1.0, 1.0]
    assert list(tri.get_ultimate_losses()) == [0.0, 0.0, 5.0]


def test_random_triangles_python_and_cpp_agree():
    rng = np.random.default_rng(99)
    for _ in range(30):
        n = int(rng.integers(2, 12))
        first = rng.uniform(100, 1e6, n)
        rows = []
        for i in range(n):
            row = [float(first[i])]
            for _ in range(n - i - 1):
                row.append(row[-1] * float(rng.uniform(1.0, 2.5)))
            rows.append(row)
        cl = reserving.chain_ladder(rows)
        tri = LossTriangle()
        for i, row in enumerate(rows):
            tri.add_origin_year(2000 + i, row)
        assert np.allclose(tri.get_development_factors(), cl.factors, rtol=1e-12)
        assert np.allclose(tri.get_ultimate_losses(), cl.ultimate, rtol=1e-10)
        assert sum(tri.get_ibnr_reserves()) == pytest.approx(cl.total_reserve, rel=1e-10)


def test_taylor_ashe_chain_ladder_and_mack():
    tri = reserving.genins()
    assert reserving.chain_ladder(tri).total_reserve == pytest.approx(18_680_856, abs=1)
    assert reserving.mack_chain_ladder(tri).total_se == pytest.approx(2_447_095, abs=1)


# ---------------------------------------------------------------------------
# 8. NaN no longer slips through `x < 0` checks
# ---------------------------------------------------------------------------


@needs_053_core
@pytest.mark.parametrize("call", [
    lambda: ExperienceRating(NAN, 0.0),
    lambda: ExperienceRating(1082.0, NAN),
    lambda: ExperienceRating(1082.0, 0.0).buhlmann_straub_credibility(0.0, 0.0),
    lambda: ExperienceRating(1082.0, 0.0).buhlmann_straub_credibility(NAN, 1.0),
    lambda: ExperienceRating(1082.0, 0.0).experience_mod_factor(NAN, 1000.0),
    lambda: ExperienceRating(1082.0, 0.0).experience_mod_factor(500.0, NAN),
    lambda: ExperienceRating(1082.0, 0.0).credibility_weighted_rate(1.0, 1.0, NAN),
    lambda: RateAnalyzer(1.0, 0.3).on_level_factor([0.05, NAN]),
    lambda: RateAnalyzer(1.0, 0.3).trend_factor(NAN, 2.0),
    lambda: RateAnalyzer(1.0, 0.3).trend_factor(0.05, NAN),
    lambda: RateAnalyzer(1.0, 0.3).loss_cost_rate(NAN, 10.0),
    lambda: RateAnalyzer(1.0, 0.3).loss_cost_rate(10.0, NAN),
    lambda: RateAnalyzer(1.0, 0.3).required_rate_change(NAN),
    lambda: ActuarialMath.calculate_loss_ratio(NAN, 100.0),
    lambda: ActuarialMath.calculate_loss_ratio(10.0, NAN),
    lambda: ActuarialMath.present_value(NAN, 5, 100.0),
    lambda: ExposureRating(1000.0, 0.5).burning_cost([100.0, NAN], 0.0, 1000.0, 5),
    lambda: MonteCarloSimulator(5, 1).simulate_economic_path(100.0, NAN, 0.2, 5),
    lambda: MonteCarloSimulator(5, 1).simulate_economic_path(INF, 0.0, 0.2, 5),
    lambda: MonteCarloSimulator(5, 1).simulate_life_portfolio(10, 0.1, 0.1, INF),
    lambda: MonteCarloSimulator(5, 1).simulate_catastrophe_loss(0.5, INF, 2.0),
])
def test_nan_and_inf_inputs_rejected(call):
    with pytest.raises(ValueError):
        call()


# ---------------------------------------------------------------------------
# 4. Expected Shortfall on distributions with atoms
# ---------------------------------------------------------------------------


def _es_reference(values, level):
    """Mean of the worst (1 - level) share, splitting the boundary atom."""
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    n = x.size
    m = n * (1.0 - level)
    whole = int(math.floor(m + 1e-9))
    total = x[:whole].sum() + (m - whole) * (x[whole] if whole < n else 0.0)
    return total / m


def test_es_on_atom_is_acerbi_tasche():
    # Used to be 100 (the mean of every value >= VaR = 0); the true ES95 is 0.01·100/0.05 = 20.
    res = mc.simulate(lambda ev, sev: ev * sev, trials=200_000, seed=1, ev=mc.Bernoulli(0.01), sev=mc.Constant(100.0))
    assert res.var(0.95) == 0.0
    assert res.tvar(0.95) == pytest.approx(res.mean / 0.05, rel=1e-12)
    assert res.tvar(0.95) == pytest.approx(20.0, rel=0.05)


@pytest.mark.parametrize("level", [0.5, 0.9, 0.95, 0.99, 0.995])
def test_es_formula_and_ordering_with_atoms(level):
    # A mixture with point masses at 0 and at 50 plus a continuous tail.
    res = mc.simulate(
        lambda a, b, c: np.where(a < 0.6, 0.0, np.where(a < 0.97, 50.0, 50.0 + c)),
        trials=40_000, seed=3, a=mc.Uniform(0, 1), b=mc.Normal(0, 1), c=mc.Exponential(0.01),
    )
    es = res.tvar(level)
    assert es >= res.var(level) - 1e-12
    assert es == pytest.approx(_es_reference(res.values, level), rel=1e-9)
    assert es <= res.values.max()


def test_es_monotone_in_level():
    res = mc.simulate(lambda p: p, trials=10_000, seed=4, p=mc.Poisson(2.0))
    levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    es = [res.tvar(l) for l in levels]
    assert all(b >= a - 1e-12 for a, b in zip(es, es[1:]))
    for l, e in zip(levels, es):
        assert e == pytest.approx(_es_reference(res.values, l), rel=1e-9)


# ---------------------------------------------------------------------------
# 9. Gamma / Beta quantiles stay monotone for tiny shapes
# ---------------------------------------------------------------------------


QS = np.array([1e-300, 1e-100, 1e-12, 1e-6, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])


@pytest.mark.parametrize("shape", [0.002, 0.01, 0.05, 0.3])
def test_gamma_ppf_monotone_tiny_shape(shape):
    x = np.asarray(mc.Gamma(shape, 2.0).ppf(QS))
    assert np.all(np.diff(x) >= 0.0)
    assert mc.Gamma(0.01, 1.0).ppf(1e-6) <= mc.Gamma(0.01, 1.0).ppf(0.5)


@pytest.mark.parametrize("a,b", [(0.01, 0.01), (0.01, 2.0), (0.005, 0.5), (0.05, 5.0)])
def test_beta_ppf_monotone_tiny_shape(a, b):
    x = np.asarray(mc.Beta(a, b).ppf(QS))
    assert np.all(np.diff(x) >= 0.0)
    assert np.all((x >= 0.0) & (x <= 1.0))


def test_gamma_ppf_tiny_shape_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for shape in [0.01, 0.05]:
        q = np.array([1e-12, 1e-6, 1e-3, 0.5])
        ref = stats.gamma.ppf(q, shape)
        got = np.asarray(mc.Gamma(shape, 1.0).ppf(q))
        ok = ref > 1e-300
        assert np.allclose(got[ok], ref[ok], rtol=1e-8)
        assert np.all(got[~ok] <= 1e-300)


# ---------------------------------------------------------------------------
# 5. bootstrap with φ = 0;  10. argument types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("process", ["odp", "gamma", "none"])
def test_bootstrap_proportional_triangle_is_deterministic(process):
    tri = [[10, 20, 30, 40], [20, 40, 60], [30, 60], [40]]
    boot = reserving.bootstrap_chain_ladder(tri, n=50, seed=1, process=process)
    cl = reserving.chain_ladder(tri)
    assert boot.phi == 0.0
    assert np.all(np.isfinite(boot.total))
    assert np.allclose(boot.total, cl.total_reserve)


def test_per_origin_accepts_numpy_scalars():
    tri = reserving.genins()
    plain = reserving.bornhuetter_ferguson(tri, 5_000_000, 0.6).total_reserve
    assert reserving.bornhuetter_ferguson(tri, np.int64(5_000_000), np.float64(0.6)).total_reserve == pytest.approx(plain)
    assert reserving.bornhuetter_ferguson(tri, np.float32(5_000_000), 0.6).total_reserve == pytest.approx(plain, rel=1e-6)


def test_bools_rejected():
    tri = reserving.genins()
    with pytest.raises(InputError):
        reserving.chain_ladder(tri, n_periods=True)
    with pytest.raises(InputError):
        reserving.chain_ladder(tri, tail=True)
    with pytest.raises(InputError):
        reserving.bornhuetter_ferguson(tri, True, 0.6)
    assert reserving.chain_ladder(tri, n_periods=np.int64(3)).factors.size == tri.n_dev - 1


def test_triangle_dataframe_with_strings_is_input_error():
    pd = pytest.importorskip("pandas")
    with pytest.raises(InputError):
        reserving.Triangle(pd.DataFrame([["a", 1], [2, 3]]))
    with pytest.raises(InputError):
        reserving.Triangle(np.array([["a", "1"], ["2", "3"]], dtype=object))
