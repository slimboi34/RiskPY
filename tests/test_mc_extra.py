"""Tests for the 0.3 additions to the Monte Carlo engine.

The analytic layer on every distribution, Latin hypercube sampling, rank
correlation by Iman–Conover, the composite distributions, the copula hook, and
the new Result methods. Statistical assertions use fixed seeds and generous
tolerances, as in test_mc.py.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from riskpy import mc  # noqa: E402
from riskpy.mc import (  # noqa: E402
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Constant,
    Empirical,
    Exponential,
    Gamma,
    LogNormal,
    Mixture,
    Model,
    NegativeBinomial,
    Normal,
    Pareto,
    PERT,
    Poisson,
    StudentT,
    Triangular,
    Truncated,
    Uniform,
    Weibull,
    correlation_matrix,
    iman_conover,
    simulate,
)

SEED = 4242

CONTINUOUS = [
    Normal(2.0, 3.0),
    LogNormal(1.2, 0.7),
    Uniform(-1.0, 4.0),
    Triangular(0.0, 1.0, 5.0),
    PERT(0.0, 2.0, 10.0),
    Exponential(3.0),
    Gamma(2.5, 1.7),
    Gamma(0.4, 2.0),
    Beta(2.0, 5.0),
    Beta(0.5, 0.5),
    Pareto(2.0, 3.0),
    Weibull(1.5, 2.0),
    StudentT(4.0, 1.0, 2.0),
]

DISCRETE = [Poisson(7.5), NegativeBinomial(3.2, 0.3), Binomial(20, 0.35), Bernoulli(0.3)]


# ---------------------------------------------------------------------------
# Analytic layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dist", CONTINUOUS, ids=lambda d: repr(d))
def test_cdf_inverts_ppf(dist):
    q = np.array([1e-5, 0.001, 0.05, 0.5, 0.95, 0.999, 1 - 1e-5])
    assert np.max(np.abs(dist.cdf(dist.ppf(q)) - q)) < 1e-9


@pytest.mark.parametrize("dist", CONTINUOUS, ids=lambda d: repr(d))
def test_density_integrates_to_the_cdf(dist):
    """Simpson's rule on the pdf between two quantiles must give their CDF gap."""
    lo, hi = float(dist.ppf(0.05)), float(dist.ppf(0.95))
    grid = np.linspace(lo, hi, 40_001)
    density = dist.pdf(grid)
    weights = np.ones_like(grid)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    integral = float((weights * density).sum() * (grid[1] - grid[0]) / 3.0)
    assert integral == pytest.approx(0.9, abs=2e-5)


@pytest.mark.parametrize("dist", CONTINUOUS + DISCRETE, ids=lambda d: repr(d))
def test_sample_matches_analytic_moments(dist):
    rng = np.random.default_rng(SEED)
    m = dist.moments()
    sample = dist.sample(200_000, rng)
    assert float(sample.mean()) == pytest.approx(m.mean, abs=5.0 * m.sd / math.sqrt(200_000))
    assert float(sample.std()) == pytest.approx(m.sd, rel=0.06)


@pytest.mark.parametrize("dist", DISCRETE, ids=lambda d: repr(d))
def test_discrete_pmf_sums_to_one_and_cdf_is_a_step(dist):
    k = np.arange(0, 200)
    assert dist.pmf(k).sum() == pytest.approx(1.0, abs=1e-12)
    assert dist.pdf(np.array([2.5])) == 0.0  # no mass between integers
    assert float(dist.cdf(3.0)) == pytest.approx(float(dist.pmf(np.arange(4)).sum()), abs=1e-12)
    assert float(dist.cdf(3.9)) == float(dist.cdf(3.0))
    q = np.array([0.01, 0.3, 0.7, 0.99])
    x = dist.ppf(q)
    assert np.all(dist.cdf(x) >= q - 1e-12)
    assert np.all(dist.cdf(x - 1.0) < q)


def test_analytic_layer_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    pairs = [
        (Normal(2, 3), stats.norm(2, 3)),
        (LogNormal(1.2, 0.7), stats.lognorm(0.7, scale=math.exp(1.2))),
        (Gamma(2.5, 1.7), stats.gamma(2.5, scale=1.7)),
        (Beta(2.0, 5.0), stats.beta(2.0, 5.0)),
        (Beta(0.2, 8.0), stats.beta(0.2, 8.0)),
        (Weibull(1.5, 2.0), stats.weibull_min(1.5, scale=2.0)),
        (StudentT(4.0, 1.0, 2.0), stats.t(4.0, loc=1.0, scale=2.0)),
        (StudentT(1.5), stats.t(1.5)),
        (Pareto(2.0, 3.0), stats.pareto(3.0, scale=2.0)),
        (Triangular(0, 1, 5), stats.triang(0.2, loc=0, scale=5)),
        (PERT(0, 2, 10), stats.beta(1.8, 4.2, loc=0, scale=10)),
    ]
    q = np.linspace(1e-6, 1 - 1e-6, 999)
    for dist, ref in pairs:
        x = np.linspace(ref.ppf(0.001), ref.ppf(0.999), 501)
        assert np.max(np.abs(dist.pdf(x) - ref.pdf(x))) < 1e-12 * max(1.0, float(np.max(ref.pdf(x))))
        assert np.max(np.abs(dist.cdf(x) - ref.cdf(x))) < 1e-13
        truth = ref.ppf(q)
        # Relative where the quantile is large, absolute where it crosses zero.
        # SciPy's own t.ppf is 6e-8 off at the median (its beta inversion
        # loses 1 - z there; ours inverts the complementary beta instead), so
        # the t family gets the tolerance SciPy can meet, not the one we can.
        tolerance = 1e-7 if isinstance(dist, StudentT) else 1e-10
        assert np.max(np.abs(dist.ppf(q) - truth) / np.maximum(1.0, np.abs(truth))) < tolerance
        if isinstance(dist, StudentT):
            assert dist.ppf(0.5) == dist.loc  # exactly, which SciPy does not manage
        assert dist.moments().mean == pytest.approx(float(ref.mean()), rel=1e-12)
        assert dist.moments().variance == pytest.approx(float(ref.var()), rel=1e-12)
    for dist, ref in [(Poisson(7.5), stats.poisson(7.5)), (NegativeBinomial(3.2, 0.3), stats.nbinom(3.2, 0.3)),
                      (Binomial(20, 0.35), stats.binom(20, 0.35))]:
        k = np.arange(0, 60)
        assert np.max(np.abs(dist.pmf(k) - ref.pmf(k))) < 1e-14
        assert np.max(np.abs(dist.ppf(np.linspace(0.01, 0.99, 99)) - ref.ppf(np.linspace(0.01, 0.99, 99)))) == 0.0


def test_scalar_in_scalar_out():
    assert isinstance(Normal(0, 1).ppf(0.975), float)
    assert Normal(0, 1).ppf(0.975) == pytest.approx(1.959963984540054)
    assert isinstance(Gamma(2, 3).ppf(0.5), float)
    assert float(Normal(0, 1).cdf(1.96)) == pytest.approx(0.9750021048517795)


def test_moments_report_infinite_where_they_are():
    assert Pareto(1.0, 0.8).moments().mean == math.inf
    assert Pareto(1.0, 1.5).moments().variance == math.inf
    assert math.isfinite(Pareto(1.0, 2.5).moments().variance)
    assert StudentT(1.5).moments().variance == math.inf
    assert StudentT(0.9).moments().mean == math.inf
    assert Constant(3.0).moments().cv == math.inf or Constant(3.0).moments().sd == 0.0


def test_alternative_constructors():
    ln = LogNormal.from_median_cv(median=1000.0, cv=0.8)
    assert math.exp(ln.mu) == pytest.approx(1000.0)
    assert ln.moments().cv == pytest.approx(0.8, rel=1e-12)
    g = Gamma.from_moments(mean=12.0, sd=4.0)
    assert g.moments().mean == pytest.approx(12.0) and g.moments().sd == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Composite distributions
# ---------------------------------------------------------------------------


def test_truncated_normal_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    tr = Truncated(Normal(0.0, 1.0), low=-1.0, high=2.0)
    ref = stats.truncnorm(-1.0, 2.0)
    x = np.linspace(-1.0, 2.0, 301)
    assert np.max(np.abs(tr.cdf(x) - ref.cdf(x))) < 1e-14
    assert np.max(np.abs(tr.pdf(x) - ref.pdf(x))) < 1e-13
    assert tr.moments().mean == pytest.approx(float(ref.mean()), abs=1e-7)
    assert tr.moments().variance == pytest.approx(float(ref.var()), abs=1e-7)
    sample = tr.sample(50_000, np.random.default_rng(SEED))
    assert sample.min() >= -1.0 and sample.max() <= 2.0
    assert sample.mean() == pytest.approx(float(ref.mean()), abs=0.02)


def test_truncated_discrete_and_one_sided():
    tr = Truncated(Poisson(5.0), low=2)
    assert tr.pdf(np.arange(0, 80)).sum() == pytest.approx(1.0, abs=1e-12)
    assert tr.pdf(np.array([1.0]))[0] == 0.0
    assert tr.sample(20_000, np.random.default_rng(SEED)).min() >= 2.0
    assert math.isfinite(tr.moments().mean)
    capped = Truncated(LogNormal(8.0, 1.0), high=10_000.0)
    assert capped.sample(20_000, np.random.default_rng(SEED)).max() <= 10_000.0


def test_truncated_rejects_nonsense():
    with pytest.raises(ValueError):
        Truncated(Normal(0, 1))
    with pytest.raises(ValueError):
        Truncated(Normal(0, 1), low=3.0, high=1.0)
    with pytest.raises(ValueError):
        Truncated(Uniform(0, 1), low=5.0, high=6.0)  # no mass there
    with pytest.raises(TypeError):
        Truncated(3.0, low=0.0)


def test_mixture_moments_and_sampling():
    mix = Mixture([LogNormal(8.0, 1.0), Pareto(50_000.0, 2.5)], [0.9, 0.1])
    parts = [c.moments() for c in mix.components]
    expected_mean = 0.9 * parts[0].mean + 0.1 * parts[1].mean
    assert mix.moments().mean == pytest.approx(expected_mean, rel=1e-12)
    grid = np.linspace(0.0, 2e5, 200_001)
    assert np.max(np.abs(mix.cdf(grid) - (0.9 * mix.components[0].cdf(grid) + 0.1 * mix.components[1].cdf(grid)))) < 1e-15
    sample = mix.sample(200_000, np.random.default_rng(SEED))
    assert sample.mean() == pytest.approx(expected_mean, rel=0.05)
    assert Mixture([Poisson(2), Poisson(9)]).discrete is True
    with pytest.raises(ValueError):
        Mixture([])
    with pytest.raises(ValueError):
        Mixture([Normal(0, 1)], [0.0])
    with pytest.raises(TypeError):
        Mixture([Normal(0, 1), 2.0])


def test_categorical_and_empirical():
    cat = Categorical([10.0, 20.0, 50.0], [0.5, 0.3, 0.2])
    assert cat.moments().mean == 21.0
    assert list(cat.ppf(np.array([0.1, 0.6, 0.95]))) == [10.0, 20.0, 50.0]
    assert list(cat.cdf(np.array([9.0, 10.0, 25.0, 50.0]))) == [0.0, 0.5, 0.8, 1.0]
    assert cat.pmf(np.array([20.0]))[0] == 0.3
    sample = cat.sample(50_000, np.random.default_rng(SEED))
    assert set(np.unique(sample)) == {10.0, 20.0, 50.0}
    with pytest.raises(ValueError):
        Categorical([1.0, 2.0], [0.5])
    with pytest.raises(ValueError):
        Categorical([1.0, 2.0], [-1.0, 2.0])

    emp = Empirical([1.0, 2.0, 2.0, 5.0, 9.0])
    assert list(emp.cdf(np.array([0.0, 2.0, 5.0, 9.0]))) == [0.0, 0.6, 0.8, 1.0]
    assert list(emp.ppf(np.array([0.0, 0.2, 0.5, 1.0]))) == [1.0, 1.0, 2.0, 9.0]
    with pytest.raises(ValueError):
        Empirical([1.0, float("nan")])


# ---------------------------------------------------------------------------
# Latin hypercube sampling
# ---------------------------------------------------------------------------


def test_lhs_keeps_the_marginal_and_cuts_the_noise():
    dist = LogNormal.from_moments(18_000.0, 42_000.0)
    m = dist.moments()
    rng = np.random.default_rng(SEED)
    lhs = dist.sample_lhs(100_000, rng)
    # Exactly one draw per stratum, so the empirical CDF is nearly perfect.
    assert np.max(np.abs(np.sort(dist.cdf(lhs)) - (np.arange(1, 100_001) - 0.5) / 100_000)) < 1e-5 + 1.0 / 100_000
    assert lhs.mean() == pytest.approx(m.mean, abs=0.5 * m.sd / math.sqrt(100_000))

    def spread(sampling):
        return np.std([simulate(lambda a, b: a * b, trials=2_000, seed=s, sampling=sampling,
                                a=LogNormal(8, 1), b=Gamma(2, 3)).mean for s in range(40)])

    assert spread("random") / spread("lhs") > 1.4


def test_lhs_on_a_constant_and_on_a_discrete():
    rng = np.random.default_rng(SEED)
    assert np.all(Constant(2.0).sample_lhs(100, rng) == 2.0)
    counts = Poisson(140.0).sample_lhs(100_000, rng)
    assert counts.mean() == pytest.approx(140.0, abs=0.05)
    assert np.all(counts == np.round(counts))


def test_bad_sampling_name_is_rejected():
    with pytest.raises(ValueError, match="sampling"):
        simulate(lambda x: x, trials=10, sampling="sobol", x=Normal(0, 1))


# ---------------------------------------------------------------------------
# Dependence
# ---------------------------------------------------------------------------


def _spearman(x, y):
    rx, ry = mc._rankdata(np.asarray(x), np), mc._rankdata(np.asarray(y), np)
    return float(np.corrcoef(rx, ry)[0, 1])


def test_correlate_recovers_the_rank_correlation_and_keeps_marginals():
    model = Model(a=LogNormal(8, 1), b=Gamma(2, 3), c=Poisson(30))
    model.correlate("a", "b", 0.7).correlate("a", "c", -0.4)
    for sampling in ("random", "lhs"):
        drawn = model.sample(100_000, seed=SEED, sampling=sampling)
        assert _spearman(drawn["a"], drawn["b"]) == pytest.approx(0.7, abs=0.01)
        assert _spearman(drawn["a"], drawn["c"]) == pytest.approx(-0.4, abs=0.01)
        assert _spearman(drawn["b"], drawn["c"]) == pytest.approx(0.0, abs=0.01)
    independent = Model(a=LogNormal(8, 1), b=Gamma(2, 3), c=Poisson(30)).sample(100_000, seed=SEED)
    dependent = model.sample(100_000, seed=SEED)
    # Same draws, only re-paired: the sorted columns are identical.
    assert np.array_equal(np.sort(independent["a"]), np.sort(dependent["a"]))


def test_correlation_accepts_a_matrix_and_validates_it():
    matrix = correlation_matrix(["x", "y"], {("x", "y"): 0.5})
    result = simulate(lambda x, y: x + y, trials=50_000, seed=SEED, correlation=matrix, x=Normal(0, 1), y=Normal(0, 1))
    assert _spearman(result.inputs["x"], result.inputs["y"]) == pytest.approx(0.5, abs=0.015)
    with pytest.raises(ValueError, match="unknown variable"):
        correlation_matrix(["x", "y"], {("x", "z"): 0.5})
    with pytest.raises(ValueError, match="itself"):
        correlation_matrix(["x", "y"], {("x", "x"): 0.5})
    with pytest.raises(ValueError, match="\\[-1, 1\\]"):
        correlation_matrix(["x", "y"], {("x", "y"): 1.5})
    with pytest.raises(ValueError, match="positive semi-definite"):
        Model(a=Normal(0, 1), b=Normal(0, 1), c=Normal(0, 1)).correlate("a", "b", 0.9).correlate("a", "c", 0.9).correlate("b", "c", -0.9)
    with pytest.raises(ValueError, match="unknown variable"):
        Model(a=Normal(0, 1)).correlate("a", "q", 0.1)
    with pytest.raises(ValueError):
        simulate(lambda x, y: x + y, trials=10, correlation=np.eye(3), x=Normal(0, 1), y=Normal(0, 1))


def test_iman_conover_pearson_flag():
    rng = np.random.default_rng(SEED)
    samples = {"x": rng.normal(size=50_000), "y": rng.normal(size=50_000)}
    target = np.array([[1.0, 0.6], [0.6, 1.0]])
    out = iman_conover(samples, target, rng, spearman=False)
    assert float(np.corrcoef(out["x"], out["y"])[0, 1]) == pytest.approx(0.6, abs=0.015)
    with pytest.raises(ValueError):
        iman_conover(samples, np.eye(3), rng)


class _ComonotoneCopula:
    """Every marginal drawn at the same uniform — perfect rank dependence."""

    dim = 2

    def uniforms(self, n, rng):
        u = rng.random(n)
        return np.column_stack([u, u])


def test_copula_protocol_drives_the_draws():
    result = simulate(lambda a, b: a + b, trials=20_000, seed=SEED, copula=_ComonotoneCopula(),
                      a=LogNormal(8, 1), b=Gamma(2, 3))
    assert _spearman(result.inputs["a"], result.inputs["b"]) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError, match="either correlation"):
        simulate(lambda a, b: a + b, trials=10, copula=_ComonotoneCopula(), correlation={("a", "b"): 0.1},
                 a=Normal(0, 1), b=Normal(0, 1))
    with pytest.raises(ValueError, match="dimension"):
        simulate(lambda a: a, trials=10, copula=_ComonotoneCopula(), a=Normal(0, 1))


# ---------------------------------------------------------------------------
# Result and Model additions
# ---------------------------------------------------------------------------


def test_result_summary_statistics():
    result = simulate(lambda x: x, trials=100_000, seed=SEED, x=LogNormal(0.0, 1.0))
    assert result.median == pytest.approx(1.0, abs=0.03)
    assert result.skewness > 3.0  # log-normal with σ = 1 is very right-skewed
    assert result.kurtosis > 10.0
    assert result.cdf(result.var(0.9)) == pytest.approx(0.9, abs=1e-4)
    assert result.percentile(0.0) == float(result.values.min())
    assert result.percentile(100.0) == float(result.values.max())
    for q in (1.0, 37.5, 99.9):
        assert result.percentile(q) == pytest.approx(float(np.percentile(result.values, q)), rel=1e-12)
    assert set(result.percentiles([50, 99])) == {50.0, 99.0}
    edges, counts = result.histogram(30)
    assert len(edges) == 31 and counts.sum() == 100_000
    x, survival = result.exceedance_curve()
    assert np.all(np.diff(x) >= 0) and np.all(np.diff(survival) < 0)
    described = result.describe()
    assert described["trials"] == 100_000 and described["var_995"] >= described["var_99"]
    assert len(result) == 100_000
    with pytest.raises(ValueError):
        result.percentile(101.0)


def test_tvar_uses_every_tail_value():
    result = simulate(lambda x: x, trials=20_000, seed=SEED, x=Normal(0, 1))
    threshold = result.var(0.99)
    assert result.tvar(0.99) == pytest.approx(float(result.values[result.values >= threshold].mean()), rel=1e-12)


def test_sensitivity_methods():
    result = simulate(lambda big, small: 10.0 * big + small, trials=30_000, seed=SEED, big=Normal(0, 1), small=Normal(0, 1))
    pearson = dict(result.sensitivity("pearson"))
    contribution = dict(result.sensitivity("contribution"))
    assert pearson["big"] > 0.99
    assert contribution["big"] + contribution["small"] == pytest.approx(1.0)
    assert contribution["big"] > 0.98
    with pytest.raises(ValueError):
        result.sensitivity("kendall")


def test_model_sweep_and_sample():
    model = Model(a=Normal(10, 1), b=Constant(2.0))
    model.formula(lambda a, b: a * b)
    scenarios = model.sweep("b", {"low": Constant(1.0), "high": Constant(3.0)}, trials=2_000, seed=SEED)
    assert set(scenarios) == {"low", "high"}
    assert scenarios["low"].label == "low"
    assert scenarios["high"].mean == pytest.approx(3.0 * scenarios["low"].mean, rel=1e-9)
    assert model.variables["b"].value == 2.0  # restored afterwards
    with pytest.raises(ValueError):
        model.sweep("zzz", {"x": Constant(1.0)})
    with pytest.raises(TypeError):
        model.sweep("b", {"x": 1.0})
    drawn = model.sample(500, seed=SEED)
    assert set(drawn) == {"a", "b"} and drawn["a"].shape == (500,)


def test_plot_kinds_are_validated():
    result = simulate(lambda x: x, trials=100, seed=SEED, x=Normal(0, 1))
    with pytest.raises(ValueError, match="kind"):
        result.plot("pie")


def test_plain_number_rejected_in_simulate_too():
    with pytest.raises(TypeError, match="Constant"):
        simulate(lambda x: x, trials=10, x=3.0)
