"""riskpy.capital: copulas, aggregation and allocation, checked against closed forms.

Every reference here is computed by a route the library has no hand in: a textbook formula
worked in the test, SciPy's distributions, a published matrix, or arithmetic done by hand
on a sample small enough to read.
"""
import math

import pytest

np = pytest.importorskip("numpy")

from riskpy import capital, tools  # noqa: E402
from riskpy.capital import (  # noqa: E402
    ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula, StudentTCopula, TailDependence, aggregate_capital,
    allocate, empirical_tail_dependence, nearest_correlation,
)
from riskpy.errors import InputError, RiskPyError  # noqa: E402
from riskpy.mc import LogNormal, Model, Normal  # noqa: E402


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Closed forms


def test_kendall_tau_closed_forms():
    assert GaussianCopula(0.5).kendall_tau() == pytest.approx(1.0 / 3.0, abs=1e-15)      # 2/π·asin(½) = 2/π·π/6
    assert StudentTCopula(0.5, df=3).kendall_tau() == pytest.approx(1.0 / 3.0, abs=1e-15)
    assert GaussianCopula(-1.0).kendall_tau() == pytest.approx(-1.0)
    assert ClaytonCopula(2.0).kendall_tau() == pytest.approx(0.5)                        # θ/(θ+2)
    assert ClaytonCopula(6.0).kendall_tau() == pytest.approx(0.75)
    assert GumbelCopula(4.0).kendall_tau() == pytest.approx(0.75)                        # 1 − 1/θ
    assert GumbelCopula(1.0).kendall_tau() == 0.0


def test_spearman_rho_of_the_gaussian_copula():
    # 6/π·asin(ρ/2); at ρ = 1 it is 6/π·π/6 = 1, and it is what Model.correlate inverts.
    assert GaussianCopula(1.0).spearman_rho() == pytest.approx(1.0)
    rho_s = GaussianCopula(0.5).spearman_rho()
    assert rho_s == pytest.approx(6.0 / math.pi * math.asin(0.25), abs=1e-15)
    assert 2.0 * math.sin(math.pi * rho_s / 6.0) == pytest.approx(0.5, abs=1e-14)


def test_frank_tau_against_the_debye_integral():
    integrate = pytest.importorskip("scipy.integrate")
    for theta in (0.5, 2.0, 5.736, 20.0):
        debye = integrate.quad(lambda t: t / math.expm1(t), 0.0, theta)[0] / theta
        assert FrankCopula(theta).kendall_tau() == pytest.approx(1.0 - 4.0 / theta * (1.0 - debye), abs=1e-9)
    # tau is odd in theta, and the textbook value theta = 5.736 gives tau = 0.5
    assert FrankCopula(-3.0).kendall_tau() == pytest.approx(-FrankCopula(3.0).kendall_tau(), abs=1e-10)
    assert FrankCopula(5.736).kendall_tau() == pytest.approx(0.5, abs=1e-4)


def test_tail_dependence_closed_forms():
    stats = pytest.importorskip("scipy.stats")
    assert GaussianCopula(0.99).tail_dependence() == TailDependence(0.0, 0.0)
    assert GaussianCopula(1.0).tail_dependence() == TailDependence(1.0, 1.0)
    for rho, df in ((0.5, 4.0), (0.0, 3.0), (-0.4, 10.0), (0.9, 2.5)):
        reference = 2.0 * stats.t.cdf(-math.sqrt((df + 1.0) * (1.0 - rho) / (1.0 + rho)), df + 1.0)
        tails = StudentTCopula(rho, df).tail_dependence()
        assert tails.lower == tails.upper == pytest.approx(reference, abs=1e-10)
    # df = 1, rho = 0: t₂(−√2) = ½ − √2/(2·2), so λ = 1 − 1/√2
    assert StudentTCopula(0.0, 1.0).tail_dependence().upper == pytest.approx(1.0 - 1.0 / math.sqrt(2.0), abs=1e-12)
    assert ClaytonCopula(2.0).tail_dependence() == pytest.approx((2.0 ** -0.5, 0.0))
    assert GumbelCopula(2.0).tail_dependence() == pytest.approx((0.0, 2.0 - math.sqrt(2.0)))
    assert FrankCopula(8.0).tail_dependence() == TailDependence(0.0, 0.0)
    # the survival copula swaps the tails
    assert ClaytonCopula(2.0, survival=True).tail_dependence() == pytest.approx((0.0, 2.0 ** -0.5))
    assert GumbelCopula(2.0, survival=True).tail_dependence() == pytest.approx((2.0 - math.sqrt(2.0), 0.0))


def test_gaussian_copula_cdf_against_scipy_and_sheppard():
    stats = pytest.importorskip("scipy.stats")
    for rho in (-0.9, -0.3, 0.5, 0.95):
        cop = GaussianCopula(rho)
        assert cop.cdf(0.5, 0.5) == pytest.approx(0.25 + math.asin(rho) / (2.0 * math.pi), abs=1e-12)   # Sheppard
        for u, v in ((0.3, 0.6), (0.99, 0.995), (0.01, 0.9)):
            a, b = stats.norm.ppf(u), stats.norm.ppf(v)
            reference = stats.multivariate_normal(cov=[[1.0, rho], [rho, 1.0]]).cdf([a, b])
            assert cop.cdf(u, v) == pytest.approx(reference, abs=5e-7)


def test_t_copula_cdf_against_the_chi_square_mixture():
    stats = pytest.importorskip("scipy.stats")
    integrate = pytest.importorskip("scipy.integrate")

    # A bivariate t is a bivariate normal divided by √(W), W ~ χ²_df/df: integrate the normal CDF over W.
    def reference(a, b, rho, df):
        mvn = stats.multivariate_normal(cov=[[1.0, rho], [rho, 1.0]])
        f = lambda w: mvn.cdf([a * math.sqrt(w), b * math.sqrt(w)]) * stats.chi2.pdf(w * df, df) * df  # noqa: E731
        return integrate.quad(f, 0.0, np.inf, limit=200)[0]

    for rho, df, u, v in ((0.5, 4.0, 0.3, 0.6), (-0.3, 5.0, 0.9, 0.2), (0.0, 3.0, 0.95, 0.95)):
        a, b = stats.t.ppf(u, df), stats.t.ppf(v, df)
        assert StudentTCopula(rho, df).cdf(u, v) == pytest.approx(reference(a, b, rho, df), abs=2e-6)
    # Uncorrelated is not independent for a t copula: C(q, q) exceeds q² in the tail.
    assert StudentTCopula(0.0, 3.0).cdf(0.95, 0.95) > 0.95 ** 2 + 0.004
    assert StudentTCopula(-0.6, 3.0).cdf(0.5, 0.5) == pytest.approx(0.25 + math.asin(-0.6) / (2.0 * math.pi), abs=1e-10)


def test_archimedean_cdfs_are_the_textbook_formulas():
    u, v = 0.3, 0.6
    assert ClaytonCopula(2.0).cdf(u, v) == pytest.approx((u ** -2.0 + v ** -2.0 - 1.0) ** -0.5, abs=1e-14)
    assert GumbelCopula(2.0).cdf(u, v) == pytest.approx(
        math.exp(-math.sqrt(math.log(u) ** 2 + math.log(v) ** 2)), abs=1e-14)
    theta = 4.0
    frank = -math.log(1.0 + (math.exp(-theta * u) - 1.0) * (math.exp(-theta * v) - 1.0)
                      / (math.exp(-theta) - 1.0)) / theta
    assert FrankCopula(theta).cdf(u, v) == pytest.approx(frank, abs=1e-14)
    # Gumbel at theta = 1 is independence; C(u, u) = u^(2^(1/θ))
    assert GumbelCopula(1.0).cdf(u, v) == pytest.approx(u * v, abs=1e-15)
    assert GumbelCopula(3.0).cdf(0.9, 0.9) == pytest.approx(0.9 ** (2.0 ** (1.0 / 3.0)), abs=1e-14)
    # every copula: C(u, 1) = u, C(u, 0) = 0, and the Fréchet bounds
    for cop in (GaussianCopula(0.4), StudentTCopula(0.4, 5.0), ClaytonCopula(1.5), GumbelCopula(1.5),
                FrankCopula(-2.0), ClaytonCopula(1.5, survival=True)):
        assert cop.cdf(0.37, 1.0) == pytest.approx(0.37, abs=1e-12) and cop.cdf(0.0, 0.8) == 0.0
        assert max(0.3 + 0.6 - 1.0, 0.0) <= cop.cdf(0.3, 0.6) <= 0.3
    # survival Clayton: Ĉ(u, v) = u + v − 1 + C(1 − u, 1 − v)
    assert ClaytonCopula(2.0, survival=True).cdf(u, v) == pytest.approx(
        u + v - 1.0 + ((1.0 - u) ** -2.0 + (1.0 - v) ** -2.0 - 1.0) ** -0.5, abs=1e-14)


def test_joint_exceedance():
    q = 0.995
    assert GumbelCopula(2.0).joint_exceedance(q) == pytest.approx(1.0 - 2.0 * q + q ** math.sqrt(2.0), abs=1e-15)
    assert GaussianCopula(0.0).joint_exceedance(q) == pytest.approx((1.0 - q) ** 2, abs=1e-12)     # independence
    assert GaussianCopula(1.0).joint_exceedance(q) == pytest.approx(1.0 - q, abs=1e-12)            # comonotonic
    # same correlation, fatter joint tail: the t copula's whole point
    assert StudentTCopula(0.5, 4.0).joint_exceedance(q) > 2.5 * GaussianCopula(0.5).joint_exceedance(q)
    with pytest.raises(InputError) as exc:
        GumbelCopula(2.0).joint_exceedance(99.5)
    assert exc.value.code == "RP110"


def test_from_kendall_tau_round_trips():
    assert GaussianCopula.from_kendall_tau(1.0 / 3.0).correlation[0, 1] == pytest.approx(0.5, abs=1e-14)
    t = StudentTCopula.from_kendall_tau(0.5, df=6)
    assert t.kendall_tau() == pytest.approx(0.5) and t.df == 6.0
    assert ClaytonCopula.from_kendall_tau(0.5).theta == pytest.approx(2.0)
    assert GumbelCopula.from_kendall_tau(0.5).theta == pytest.approx(2.0)
    for tau in (0.5, -0.3, 0.05):
        assert FrankCopula.from_kendall_tau(tau).kendall_tau() == pytest.approx(tau, abs=1e-9)
    assert FrankCopula.from_kendall_tau(0.5).theta == pytest.approx(5.7363, abs=1e-3)   # the textbook value


# ---------------------------------------------------------------------------
# Sampling: uniform marginals, the right rank correlation, the right joint probabilities


SAMPLERS = [
    GaussianCopula(0.5), GaussianCopula(-0.7), StudentTCopula(0.5, 4.0), StudentTCopula(0.0, 3.0),
    ClaytonCopula(2.0), ClaytonCopula(2.0, survival=True), GumbelCopula(2.0), GumbelCopula(1.3, survival=True),
    FrankCopula(5.736), FrankCopula(-3.0), ClaytonCopula(1.5, dim=3), GumbelCopula(1.7, dim=4), FrankCopula(4.0, dim=3),
]


@pytest.mark.parametrize("cop", SAMPLERS, ids=repr)
def test_samples_have_uniform_marginals_and_the_closed_form_dependence(cop):
    stats = pytest.importorskip("scipy.stats")
    n = 40_000
    u = cop.sample(n, seed=20261120)
    assert u.shape == (n, cop.dim) and u.min() > 0.0 and u.max() < 1.0
    for j in range(cop.dim):
        # Kolmogorov–Smirnov against U(0,1): 1.95/√n is the 0.1% critical value; a little headroom on top
        assert stats.kstest(u[:, j], "uniform").statistic < 2.5 / math.sqrt(n)
    # Kendall's tau of the sample against the closed form (its standard error is below 0.004 here)
    last = cop.dim - 1
    assert stats.kendalltau(u[:, 0], u[:, last]).statistic == pytest.approx(cop.kendall_tau(), abs=0.015)
    # the empirical joint probability against C(u, v): binomial standard error ≈ 0.0022
    empirical = float(np.mean((u[:, 0] <= 0.3) & (u[:, last] <= 0.6)))
    assert empirical == pytest.approx(cop.cdf(0.3, 0.6), abs=0.009)


def test_samples_reproduce_the_joint_exceedance_in_the_tail():
    n, q = 400_000, 0.98
    for cop in (StudentTCopula(0.5, 4.0), GumbelCopula(2.0), ClaytonCopula(2.0, survival=True), GaussianCopula(0.5)):
        u = cop.sample(n, seed=7)
        expected = cop.joint_exceedance(q)
        observed = float(np.mean((u[:, 0] > q) & (u[:, 1] > q)))
        assert observed == pytest.approx(expected, abs=5.0 * math.sqrt(expected / n))


def test_same_seed_same_draws_and_matrix_copulas_follow_their_matrix():
    stats = pytest.importorskip("scipy.stats")
    assert np.array_equal(GumbelCopula(2.0).sample(100, seed=3), GumbelCopula(2.0).sample(100, seed=3))
    R = [[1.0, 0.6, -0.2], [0.6, 1.0, 0.1], [-0.2, 0.1, 1.0]]
    cop = GaussianCopula(R)
    assert cop.dim == 3 and cop.kendall_tau(0, 2) == pytest.approx(2.0 / math.pi * math.asin(-0.2))
    z = stats.norm.ppf(cop.sample(200_000, seed=5))
    assert np.corrcoef(z, rowvar=False) == pytest.approx(np.array(R), abs=0.01)
    # a perfectly correlated (semi-definite) matrix still samples
    u = GaussianCopula(1.0).sample(1000, seed=1)
    assert np.allclose(u[:, 0], u[:, 1])
    t3 = StudentTCopula(R, df=5)
    assert t3.tail_dependence(0, 1).upper > t3.tail_dependence(0, 2).upper > 0.0


def test_empirical_tail_dependence_sees_what_the_copula_has():
    u = GumbelCopula(2.0).sample(400_000, seed=1)
    tails = empirical_tail_dependence(u[:, 0], u[:, 1], level=0.99)
    # at a finite level the estimate is P(V > q | U > q) = joint_exceedance / (1 − q)
    assert tails.upper == pytest.approx(GumbelCopula(2.0).joint_exceedance(0.99) / 0.01, abs=0.03)
    # the lower tail has no dependence in the limit, but at 1% it is still C(0.01, 0.01)/0.01 = 0.01^(√2 − 1)
    assert tails.lower == pytest.approx(0.01 ** (math.sqrt(2.0) - 1.0), abs=0.03)
    g = GaussianCopula(0.5).sample(400_000, seed=1)
    assert empirical_tail_dependence(g[:, 0], g[:, 1], level=0.999).upper < tails.upper / 3.0
    with pytest.raises(InputError) as exc:
        empirical_tail_dependence([1.0, 2.0], [1.0], 0.9)
    assert exc.value.code == "RP109"


def test_copula_arguments_are_checked():
    cases = [
        (lambda: GaussianCopula(1.5), "RP100", "correlation"),
        (lambda: GaussianCopula([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]), "RP100", "correlation"),
        (lambda: StudentTCopula(0.5, df=0), "RP102", "df"),
        (lambda: ClaytonCopula(0.0), "RP102", "theta"),
        (lambda: GumbelCopula(0.5), "RP100", "theta"),
        (lambda: FrankCopula(0.0), "RP100", "theta"),
        (lambda: FrankCopula(-1.0, dim=3), "RP113", "theta"),
        (lambda: ClaytonCopula(2.0, dim=1), "RP107", "dim"),
        (lambda: GumbelCopula(float("nan")), "RP101", "theta"),
        (lambda: ClaytonCopula.from_kendall_tau(1.0), "RP100", "tau"),
        (lambda: GaussianCopula(0.5).cdf(1.2, 0.5), "RP103", "u"),
        (lambda: GaussianCopula(0.5).kendall_tau(0, 5), "RP100", "j"),
        (lambda: GumbelCopula(2.0).sample(0), "RP107", "n"),
    ]
    for make, code, param in cases:
        with pytest.raises(RiskPyError) as exc:
            make()
        assert (exc.value.code, exc.value.param) == (code, param), (code, param, exc.value)


# ---------------------------------------------------------------------------
# A copula inside a Model


def test_model_run_takes_a_copula_and_keeps_the_marginals():
    stats = pytest.importorskip("scipy.stats")
    model = Model(a=LogNormal(0.0, 0.5), b=Normal(10.0, 2.0))
    cop = GumbelCopula(2.0)
    draws = model.sample(60_000, seed=4, copula=cop)
    assert stats.kendalltau(draws["a"], draws["b"]).statistic == pytest.approx(0.5, abs=0.012)
    assert stats.kstest(draws["b"], stats.norm(10.0, 2.0).cdf).statistic < 1.95 / math.sqrt(60_000)
    assert float(np.mean(np.log(draws["a"]))) == pytest.approx(0.0, abs=0.01)
    with pytest.raises(InputError) as exc:
        model.sample(100, seed=1, copula=ClaytonCopula(2.0, dim=3))
    assert exc.value.code == "RP109"


def test_a_compound_has_no_quantile_function_and_is_reordered_instead():
    stats = pytest.importorskip("scipy.stats")
    from riskpy.mc import Compound, Poisson

    claims = Compound(Poisson(50), LogNormal(1.0, 0.8))
    model = Model(claims=claims, inflation=Normal(0.03, 0.01))
    draws = model.sample(40_000, seed=9, copula=StudentTCopula(0.6, 4.0))
    # ranks are the copula's, so Kendall's tau is 2/π·asin(0.6); the marginal is still the compound sum (Wald)
    assert stats.kendalltau(draws["claims"], draws["inflation"]).statistic == pytest.approx(
        2.0 / math.pi * math.asin(0.6), abs=0.015)
    mean = 50.0 * math.exp(1.0 + 0.32)
    sd = math.sqrt(50.0 * math.exp(2.0 + 2.0 * 0.64))               # λ·E[X²] for a compound Poisson
    assert float(draws["claims"].mean()) == pytest.approx(mean, abs=4.0 * sd / math.sqrt(40_000))


# ---------------------------------------------------------------------------
# nearest_correlation


def test_nearest_correlation_reproduces_highams_example():
    # Higham (2002), "Computing the nearest correlation matrix": the 3x3 example.
    fixed = nearest_correlation([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
    assert fixed == pytest.approx(np.array([[1.0, 0.7607, 0.1573], [0.7607, 1.0, 0.7607], [0.1573, 0.7607, 1.0]]),
                                  abs=5e-5)
    assert np.linalg.eigvalsh(fixed).min() > -1e-12 and np.allclose(np.diag(fixed), 1.0) and np.allclose(fixed, fixed.T)


def test_nearest_correlation_leaves_a_valid_matrix_alone_and_beats_other_candidates():
    good = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, -0.1], [0.2, -0.1, 1.0]])
    assert np.array_equal(nearest_correlation(good), good)
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(bad).min() < -0.5
    fixed = nearest_correlation(bad)
    assert np.linalg.eigvalsh(fixed).min() > -1e-10
    # No other valid correlation matrix is closer: try shrinking towards the identity, the usual hand fix.
    distance = np.linalg.norm(fixed - bad)
    for shrink in np.linspace(0.0, 1.0, 201):
        candidate = (1.0 - shrink) * bad + shrink * np.eye(3)
        if np.linalg.eigvalsh(candidate).min() >= 0.0:
            assert np.linalg.norm(candidate - bad) >= distance - 1e-9
    # the repaired matrix can be simulated from
    assert GaussianCopula(fixed).sample(10, seed=1).shape == (10, 3)
    for matrix, code in (([[1.0, 2.0], [2.0, 1.0]], "RP100"), ([[1.0, 0.5], [0.4, 1.0]], "RP100"),
                         ([[1.0, float("nan")], [float("nan"), 1.0]], "RP101"), ([1.0, 2.0], "RP100")):
        with pytest.raises(InputError) as exc:
            nearest_correlation(matrix)
        assert exc.value.code == code


# ---------------------------------------------------------------------------
# aggregate_capital: the variance–covariance formula


def test_aggregate_capital_by_hand():
    agg = aggregate_capital({"market": 100.0, "credit": 60.0}, {("market", "credit"): 0.25})
    # 100² + 60² + 2·0.25·100·60 = 16,600
    assert agg.total == pytest.approx(math.sqrt(16600.0), abs=1e-12)
    assert agg.contributions["market"] == pytest.approx((10000.0 + 1500.0) / math.sqrt(16600.0), abs=1e-12)
    assert agg.contributions["credit"] == pytest.approx((3600.0 + 1500.0) / math.sqrt(16600.0), abs=1e-12)
    assert sum(agg.contributions.values()) == pytest.approx(agg.total, abs=1e-12)
    assert agg.undiversified == 160.0
    assert agg.diversification_benefit == pytest.approx(160.0 - math.sqrt(16600.0))
    assert agg.diversification_ratio == pytest.approx(1.0 - math.sqrt(16600.0) / 160.0)
    assert sum(agg.shares.values()) == pytest.approx(1.0)
    assert "Diversification benefit" in agg.summary() and agg.to_dict()["total"] == agg.total


def test_aggregate_capital_limits_and_a_three_risk_matrix():
    # independent: Pythagoras; comonotonic: the plain sum; a perfect hedge: the difference
    assert aggregate_capital([3.0, 4.0], [[1, 0], [0, 1]]).total == pytest.approx(5.0)
    assert aggregate_capital([3.0, 4.0], [[1, 1], [1, 1]]).total == pytest.approx(7.0)
    hedge = aggregate_capital([3.0, 4.0], [[1, -1], [-1, 1]])
    assert hedge.total == pytest.approx(1.0) and hedge.contributions == pytest.approx({"risk_1": -3.0, "risk_2": 4.0})
    c = np.array([100.0, 60.0, 80.0, 25.0])
    R = np.array([[1.0, 0.5, 0.25, 0.25], [0.5, 1.0, 0.25, 0.5], [0.25, 0.25, 1.0, 0.0], [0.25, 0.5, 0.0, 1.0]])
    agg = aggregate_capital(c.tolist(), R.tolist(), names=["market", "credit", "life", "operational"])
    total = math.sqrt(float(c @ R @ c))
    assert agg.total == pytest.approx(total, rel=1e-13)
    assert list(agg.contributions.values()) == pytest.approx((c * (R @ c) / total).tolist(), rel=1e-13)
    assert agg.total < agg.undiversified
    zero = aggregate_capital([0.0, 0.0], [[1, 0.5], [0.5, 1]])
    assert zero.total == 0.0 and zero.diversification_ratio == 0.0


def test_aggregate_capital_refuses_bad_input():
    cases = [
        (lambda: aggregate_capital([], [[1.0]]), "RP111", "standalone"),
        (lambda: aggregate_capital([1.0, -2.0], [[1, 0], [0, 1]]), "RP102", "standalone"),
        (lambda: aggregate_capital([1.0, 2.0], [[1, 0.5], [0.4, 1]]), "RP100", "correlation"),
        (lambda: aggregate_capital([1.0, 2.0], [[1, 50], [50, 1]]), "RP100", "correlation"),
        (lambda: aggregate_capital([1.0, 2.0], [[1, 0]]), "RP109", "correlation"),
        (lambda: aggregate_capital([1.0, 2.0, 3.0], [[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]]), "RP100",
         "correlation"),
        (lambda: aggregate_capital({"a": 1.0, "b": 2.0}, {("a", "z"): 0.5}), "RP108", "correlation"),
        (lambda: aggregate_capital([1.0, 2.0], [[1, 0], [0, 1]], names=["x"]), "RP109", "names"),
        (lambda: aggregate_capital([1.0, "2"], [[1, 0], [0, 1]]), "RP404", "standalone"),
    ]
    for make, code, param in cases:
        with pytest.raises(RiskPyError) as exc:
            make()
        assert (exc.value.code, exc.value.param) == (code, param), (code, param, exc.value)
    # the non-PSD refusal tells the caller what to do about it
    with pytest.raises(InputError) as exc:
        aggregate_capital([1.0, 2.0, 3.0], [[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
    assert "nearest_correlation" in exc.value.hint


# ---------------------------------------------------------------------------
# allocate


def test_allocate_tvar_by_hand_on_ten_trials():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    b = np.array([5.0, 0.0, 1.0, 9.0, 2.0, 0.0, 8.0, 1.0, 3.0, 4.0])
    # totals: 6, 2, 4, 13, 7, 6, 15, 9, 12, 14. Sorted: 2 4 6 6 7 9 12 13 14 15.
    # level 0.8: the worst 20% is two trials, the totals 15 (a=7, b=8) and 14 (a=10, b=4).
    alloc = allocate({"a": a, "b": b}, level=0.8)
    assert alloc.total == pytest.approx(14.5)
    assert alloc.contributions == pytest.approx({"a": 8.5, "b": 6.0})
    # standalone: a's two worst are 10 and 9; b's are 9 and 8
    assert alloc.standalone == pytest.approx({"a": 9.5, "b": 8.5})
    assert alloc.diversification_benefit == pytest.approx(18.0 - 14.5)
    assert alloc.means == pytest.approx({"a": 5.5, "b": 3.3}) and alloc.total_mean == pytest.approx(8.8)
    assert alloc.excess == pytest.approx({"a": 3.0, "b": 2.7})
    # level 0.85: the worst 15% is 1.5 trials — all of the 15 and half of the 14.
    alloc = allocate({"a": a, "b": b}, level=0.85)
    assert alloc.total == pytest.approx((15.0 + 0.5 * 14.0) / 1.5)
    assert alloc.contributions == pytest.approx({"a": (7.0 + 0.5 * 10.0) / 1.5, "b": (8.0 + 0.5 * 4.0) / 1.5})
    # VaR at 0.8 is the 8th smallest total, 13; pro-rata to standalone VaRs (a: 8, b: 5)
    var = allocate({"a": a, "b": b}, level=0.8, measure="var", method="standalone")
    assert var.total == 13.0 and var.standalone == {"a": 8.0, "b": 5.0}
    assert var.contributions == pytest.approx({"a": 8.0, "b": 5.0})


def test_allocate_counts_only_the_part_of_an_atom_that_is_in_the_tail():
    # 100 trials; line a loses 100 in trial 0, line b loses 100 in trial 1, nothing else happens.
    a, b = np.zeros(100), np.zeros(100)
    a[0], b[1] = 100.0, 100.0
    alloc = allocate({"a": a, "b": b}, level=0.95)
    # the worst 5% is five trials: the two losses and three of the 98 zeros → ES = 200/5
    assert alloc.total == pytest.approx(40.0)
    assert alloc.contributions == pytest.approx({"a": 20.0, "b": 20.0})
    assert alloc.standalone == pytest.approx({"a": 20.0, "b": 20.0})     # each alone: 100/5
    assert alloc.diversification_benefit == pytest.approx(0.0)


def test_allocate_total_is_the_results_tvar():
    model = Model(x=LogNormal(1.0, 0.6), y=LogNormal(0.5, 0.9))
    result = model.run(50_000, seed=3, fn=lambda x, y: x + y, copula=GumbelCopula(1.5))
    for level in (0.9, 0.99, 0.995):
        alloc = allocate(result.inputs, level=level)
        assert alloc.total == pytest.approx(result.tvar(level), rel=1e-12)
        assert sum(alloc.contributions.values()) == pytest.approx(alloc.total, rel=1e-12)
        assert alloc.diversification_benefit >= 0.0                       # TVaR is subadditive


def _normal_lines(trials, seed, rho=0.5):
    u = GaussianCopula(rho).sample(trials, seed=seed)
    from scipy import stats

    return {"x": 10.0 + 2.0 * stats.norm.ppf(u[:, 0]), "y": 5.0 + 1.0 * stats.norm.ppf(u[:, 1])}


def test_euler_allocation_matches_the_normal_closed_form():
    pytest.importorskip("scipy")
    # Jointly normal lines: TVaR_i = μ_i + Cov(X_i, S)/σ_S · φ(z)/(1 − α);  VaR_i = μ_i + Cov(X_i, S)/σ_S · z.
    rho, level = 0.5, 0.99
    losses = _normal_lines(1_000_000, seed=20261120, rho=rho)
    sd_s = math.sqrt(4.0 + 1.0 + 2.0 * rho * 2.0)
    cov = {"x": 4.0 + rho * 2.0, "y": 1.0 + rho * 2.0}
    mu = {"x": 10.0, "y": 5.0}
    z = 2.3263478740408408
    ratio = phi(z) / (1.0 - level)

    tvar = allocate(losses, level=level)
    assert tvar.total == pytest.approx(15.0 + sd_s * ratio, abs=0.06)
    for name in ("x", "y"):
        assert tvar.contributions[name] == pytest.approx(mu[name] + cov[name] / sd_s * ratio, abs=0.06)
        assert tvar.standalone[name] == pytest.approx(mu[name] + math.sqrt(cov[name] - rho * 2.0) * ratio, abs=0.06)

    # the covariance method is the same thing for normal losses
    by_cov = allocate(losses, level=level, method="covariance")
    assert by_cov.total == tvar.total
    for name in ("x", "y"):
        assert by_cov.contributions[name] == pytest.approx(mu[name] + cov[name] / sd_s * ratio, abs=0.06)

    var = allocate(losses, level=level, measure="var")
    assert var.total == pytest.approx(15.0 + sd_s * z, abs=0.05)
    assert sum(var.contributions.values()) == pytest.approx(var.total, rel=1e-12)
    for name in ("x", "y"):
        assert var.contributions[name] == pytest.approx(mu[name] + cov[name] / sd_s * z, abs=0.2)


def test_allocation_of_comonotonic_and_independent_lines():
    rng = np.random.default_rng(5)
    x = rng.lognormal(0.0, 1.0, 200_000)
    # comonotonic: nothing diversifies, each line's contribution is its standalone TVaR
    alloc = allocate({"one": x, "two": 2.0 * x}, level=0.99)
    assert alloc.contributions == pytest.approx(alloc.standalone, rel=1e-12)
    assert alloc.diversification_benefit == pytest.approx(0.0, abs=1e-9)
    assert alloc.shares == pytest.approx({"one": 1.0 / 3.0, "two": 2.0 / 3.0})
    # a constant line contributes exactly its value
    alloc = allocate({"risky": x, "fixed": np.full(x.size, 7.0)}, level=0.99)
    assert alloc.contributions["fixed"] == pytest.approx(7.0) and alloc.excess["fixed"] == pytest.approx(0.0)
    # tail dependence costs capital: same marginals, same Kendall's tau, Gumbel needs more than Gaussian
    model = Model(a=LogNormal.from_moments(100, 30), b=LogNormal.from_moments(100, 30))
    gauss = allocate(model.sample(400_000, seed=1, copula=GaussianCopula.from_kendall_tau(0.5)), level=0.995)
    gumbel = allocate(model.sample(400_000, seed=1, copula=GumbelCopula.from_kendall_tau(0.5)), level=0.995)
    assert gumbel.total > gauss.total * 1.01
    assert gumbel.diversification_benefit < gauss.diversification_benefit


def test_allocate_input_forms_and_errors():
    pd = pytest.importorskip("pandas")
    matrix = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 2.0], [5.0, 0.0]])
    by_array = allocate(matrix, level=0.5, names=["a", "b"])
    by_dict = allocate({"a": matrix[:, 0], "b": matrix[:, 1]}, level=0.5)
    by_frame = allocate(pd.DataFrame(matrix, columns=["a", "b"]), level=0.5)
    assert by_array == by_dict == by_frame
    assert allocate(matrix, level=0.5).names == ("line_1", "line_2")
    frame = by_dict.to_frame()
    assert list(frame.index) == ["a", "b"] and frame["contribution"].sum() == pytest.approx(by_dict.total)
    assert "TVaR 0.5 allocation" in by_dict.summary() and by_dict.to_dict()["trials"] == 4
    cases = [
        (lambda: allocate({"a": [1.0, 2.0], "b": [1.0]}), "RP109", "losses"),
        (lambda: allocate({}, 0.9), "RP111", "losses"),
        (lambda: allocate({"a": [1.0, float("nan")]}), "RP101", "losses"),
        (lambda: allocate({"a": [1.0, 2.0]}, level=99), "RP110", "level"),
        (lambda: allocate({"a": [1.0, 2.0]}, measure="cte"), "RP108", "measure"),
        (lambda: allocate({"a": [1.0, 2.0]}, method="shapley"), "RP108", "method"),
        (lambda: allocate([1.0, 2.0, 3.0]), "RP100", "losses"),
        (lambda: allocate(matrix, names=["a"]), "RP109", "names"),
        (lambda: allocate({"a": [1.0]}), "RP111", "losses"),
    ]
    for make, code, param in cases:
        with pytest.raises(RiskPyError) as exc:
            make()
        assert (exc.value.code, exc.value.param) == (code, param), (code, param, exc.value)


# ---------------------------------------------------------------------------
# The tools: every number pinned to an independent value


def test_capital_aggregate_tool():
    out = tools.call("capital_aggregate", {"standalone": [100, 60], "correlation": [[1, 0.25], [0.25, 1]],
                                           "names": ["market", "credit"]})
    assert out["total"] == pytest.approx(math.sqrt(16600.0), abs=1e-12)
    assert out["contributions"]["market"] == pytest.approx(11500.0 / math.sqrt(16600.0), abs=1e-12)
    assert out["diversification_benefit"] == pytest.approx(160.0 - math.sqrt(16600.0), abs=1e-12)
    assert tools.TOOLS["capital_aggregate"].requires == ""          # pure Python: runs on a lean install
    err = tools.call("capital_aggregate", {"standalone": [1, 2, 3], "correlation":
                                           [[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]]}, errors="return")["error"]
    assert err["code"] == "RP100" and err["param"] == "correlation" and "nearest_correlation" in err["hint"]
    err = tools.call("capital_aggregate", {"standalone": [1, 2], "correlation": [[1, 25], [25, 1]]},
                     errors="return")["error"]
    assert err["code"] == "RP100"


def test_copula_dependence_tool():
    stats = pytest.importorskip("scipy.stats")
    out = tools.call("copula_dependence", {"family": "t", "rho": 0.5, "df": 4})
    assert out["kendall_tau"] == pytest.approx(1.0 / 3.0)
    lam = 2.0 * stats.t.cdf(-math.sqrt(5.0 * 0.5 / 1.5), 5.0)
    assert out["tail_dependence"] == pytest.approx({"lower": lam, "upper": lam}, abs=1e-10)
    assert lam == pytest.approx(0.25317, abs=1e-5)
    out = tools.call("copula_dependence", {"family": "gumbel", "copula_theta": 2.0, "level": 0.995})
    joint = 1.0 - 2.0 * 0.995 + 0.995 ** math.sqrt(2.0)
    assert out["joint_exceedance"] == pytest.approx(joint, abs=1e-15)
    assert out["multiple_of_independent"] == pytest.approx(joint / 0.005 ** 2, rel=1e-9)
    assert out["conditional_exceedance"] == pytest.approx(joint / 0.005, rel=1e-9)
    assert out["tail_dependence"]["upper"] == pytest.approx(2.0 - math.sqrt(2.0))
    out = tools.call("copula_dependence", {"family": "gaussian", "rho": 0.5})
    assert out["spearman_rho"] == pytest.approx(6.0 / math.pi * math.asin(0.25))
    assert out["tail_dependence"] == {"lower": 0.0, "upper": 0.0}
    assert tools.call("copula_dependence", {"family": "survival_clayton", "copula_theta": 2.0})[
        "tail_dependence"]["upper"] == pytest.approx(2.0 ** -0.5)
    for args, code, param in (
        ({"family": "gumbel", "rho": 0.5}, "RP113", "copula_theta"),
        ({"family": "t", "rho": 0.5}, "RP113", "df"),
        ({"family": "gaussian", "rho": 0.5, "df": 4}, "RP113", "df"),
        ({"family": "gaussian", "rho": 50}, "RP100", "rho"),
        ({"family": "clayton", "copula_theta": -1}, "RP102", "copula_theta"),
        ({"family": "gumbell", "copula_theta": 2}, "RP108", "family"),
        ({"family": "gumbel", "copula_theta": 2, "level": 99.5}, "RP110", "level"),
    ):
        err = tools.call("copula_dependence", args, errors="return")["error"]
        assert (err["code"], err["param"]) == (code, param), (args, err)


def test_nearest_correlation_tool():
    out = tools.call("nearest_correlation", {"matrix": [[1, 1, 0], [1, 1, 1], [0, 1, 1]]})
    assert out["changed"] is True and out["matrix"][0][1] == pytest.approx(0.7607, abs=5e-5)
    assert out["matrix"][0][2] == pytest.approx(0.1573, abs=5e-5)
    assert out["min_eigenvalue_before"] == pytest.approx(1.0 - math.sqrt(2.0), abs=1e-12)   # eigenvalues 1, 1 ± √2
    assert out["min_eigenvalue_after"] > -1e-12
    out = tools.call("nearest_correlation", {"matrix": [[1, 0.2], [0.2, 1]]})
    assert out["changed"] is False and out["matrix"] == [[1.0, 0.2], [0.2, 1.0]]


def test_capital_allocation_tool_matches_the_normal_closed_form():
    args = {"means": [10, 5], "sds": [2, 1], "correlation": [[1, 0.5], [0.5, 1]], "marginal": "normal",
            "level": 0.99, "trials": 1_000_000, "seed": 7, "names": ["x", "y"]}
    out = tools.call("capital_allocation", args)
    ratio = phi(2.3263478740408408) / 0.01
    sd_s = math.sqrt(7.0)
    assert out["total"] == pytest.approx(15.0 + sd_s * ratio, abs=0.06)
    assert out["contributions"]["x"] == pytest.approx(10.0 + 5.0 / sd_s * ratio, abs=0.06)
    assert out["contributions"]["y"] == pytest.approx(5.0 + 2.0 / sd_s * ratio, abs=0.06)
    assert out["standalone"]["x"] == pytest.approx(10.0 + 2.0 * ratio, abs=0.06)
    assert out["contributions"]["x"] + out["contributions"]["y"] == pytest.approx(out["total"], rel=1e-12)
    assert out["diversification_benefit"] == pytest.approx((3.0 - sd_s) * ratio, abs=0.08)
    assert tools.call("capital_allocation", args) == out               # a seed makes it reproducible


def test_capital_allocation_tool_lognormal_lines_and_refusals():
    base = {"means": [100, 60, 30], "sds": [20, 30, 10], "level": 0.995, "trials": 200_000, "seed": 3}
    gumbel = tools.call("capital_allocation", {**base, "copula": "gumbel", "copula_theta": 1.5})
    # the simulated lines have the means that were asked for (standard error of the total mean ≈ 0.1)
    assert gumbel["total_mean"] == pytest.approx(190.0, abs=0.5)
    assert gumbel["means"]["line_2"] == pytest.approx(60.0, abs=0.3)
    assert sum(gumbel["contributions"].values()) == pytest.approx(gumbel["total"], rel=1e-12)
    assert sum(gumbel["shares"].values()) == pytest.approx(1.0)
    assert gumbel["diversification_benefit"] > 0.0
    # a line's standalone TVaR is the lognormal closed form: mean · Φ(σ − z)/(1 − α)
    sigma = math.sqrt(math.log(1.0 + 0.25))
    z = 2.5758293035489004
    assert gumbel["standalone"]["line_2"] == pytest.approx(60.0 * Phi(sigma - z) / 0.005, rel=0.06)
    independent = tools.call("capital_allocation", {**base, "correlation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    assert independent["total"] < gumbel["total"]
    for extra, code, param in (
        ({"copula": "gumbel"}, "RP113", "copula_theta"),
        ({"copula": "t", "correlation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, "RP113", "df"),
        ({"correlation": [[1, 0], [0, 1]]}, "RP109", "correlation"),
        ({"sds": [20, 30]}, "RP109", "sds"),
        ({"sds": [20, 0, 10], "copula": "frank", "copula_theta": 2}, "RP102", "sds[1]"),
        ({"means": [100, -60, 30], "copula": "frank", "copula_theta": 2}, "RP102", "means[1]"),
        ({"copula": "frank", "copula_theta": 2, "trials": 10**7}, "RP405", "trials"),
        ({"copula": "frank", "copula_theta": 2, "level": 99.5}, "RP110", "level"),
        ({"copula": "frank", "copula_theta": 2, "seed": -1}, "RP102", "seed"),
        ({"copula": "clayton", "copula_theta": 2, "marginal": "gamma"}, "RP108", "marginal"),
    ):
        err = tools.call("capital_allocation", {**base, **extra}, errors="return")["error"]
        assert (err["code"], err["param"]) == (code, param), (extra, err)


def test_the_new_tools_are_described_for_an_agent():
    for name in ("capital_aggregate", "capital_allocation", "copula_dependence", "nearest_correlation"):
        explained = tools.call("riskpy_explain", {"topic": name})
        assert explained["formula"] and explained["conventions"]
        schema = tools.TOOLS[name].schema()
        assert all("description" in prop for prop in schema["properties"].values()), name
    assert capital.__doc__ and "Euler" in capital.__doc__


def test_allocation_edge_cases():
    rng = np.random.default_rng(11)
    x, y = rng.lognormal(0.0, 0.5, 50_000), rng.lognormal(0.5, 0.8, 50_000)
    losses = {"x": x, "y": y}
    # Euler VaR: a wider window averages more trials; either way the pieces add up to the VaR itself
    narrow = allocate(losses, level=0.99, measure="var", bandwidth=0.0005)
    wide = allocate(losses, level=0.99, measure="var", bandwidth=0.02)
    assert narrow.total == wide.total == float(np.sort(x + y)[49_499])       # the lower empirical quantile
    for alloc in (narrow, wide):
        assert sum(alloc.contributions.values()) == pytest.approx(alloc.total, rel=1e-12)
    assert narrow.contributions != wide.contributions
    with pytest.raises(InputError) as exc:
        allocate(losses, level=0.99, measure="var", bandwidth=2.0)
    assert (exc.value.code, exc.value.param) == ("RP100", "bandwidth")
    # every method splits the same total
    totals = {allocate(losses, level=0.99, method=m).total for m in ("euler", "covariance", "standalone")}
    assert len(totals) == 1
    pro_rata = allocate(losses, level=0.99, method="standalone")
    assert pro_rata.contributions["x"] / pro_rata.contributions["y"] == pytest.approx(
        pro_rata.standalone["x"] / pro_rata.standalone["y"], rel=1e-12)
    # nothing at risk: constant lines get their own value under every method
    flat = {"a": np.full(100, 3.0), "b": np.full(100, 5.0)}
    for method in ("euler", "covariance", "standalone"):
        alloc = allocate(flat, level=0.9, method=method)
        assert alloc.total == pytest.approx(8.0) and alloc.contributions == pytest.approx({"a": 3.0, "b": 5.0})
    zero = allocate({"a": np.zeros(50), "b": np.zeros(50)}, level=0.9, method="standalone")
    assert zero.total == 0.0 and zero.shares == {"a": 0.0, "b": 0.0}


def test_archimedean_cdfs_stay_accurate_at_extreme_theta():
    # The textbook formulas overflow or cancel at very large and very small theta; compare with the same
    # formulas evaluated in 700-digit decimal arithmetic.
    from decimal import Decimal, getcontext

    getcontext().prec = 700

    def clayton(u, v, t):
        u, v, t = Decimal(u), Decimal(v), Decimal(t)
        return float((u ** (-t) + v ** (-t) - 1) ** (-1 / t))

    def frank(u, v, t):
        u, v, t = Decimal(u), Decimal(v), Decimal(t)
        return float(-(1 + ((-t * u).exp() - 1) * ((-t * v).exp() - 1) / ((-t).exp() - 1)).ln() / t)

    points = (("0.3", "0.6"), ("0.995", "0.995"), ("0.01", "0.9"), ("0.001", "0.002"))
    for theta in ("1e-9", "1e-4", "0.5", "29", "31", "700", "5000"):
        for u, v in points:
            assert ClaytonCopula(float(theta)).cdf(float(u), float(v)) == pytest.approx(clayton(u, v, theta), abs=1e-14)
    for theta in ("1e-9", "0.5", "0.99", "1.01", "20", "499", "-1e-9", "-0.99", "-1.01", "-50", "-499"):
        for u, v in points:
            assert FrankCopula(float(theta)).cdf(float(u), float(v)) == pytest.approx(frank(u, v, theta), abs=1e-14)
    # near-perfect dependence: both exceed together almost always, or (negative Frank) almost never. 0.8.0 returned
    # exactly 0.0 for the Frank case, 1 - 2q + C(q, q) cancelled; 0.9.0 keeps the digits (600-digit mpmath value).
    assert ClaytonCopula(1e6, survival=True).joint_exceedance(0.995) == pytest.approx(0.005, rel=1e-3)
    assert FrankCopula(-499.0).joint_exceedance(0.995) == pytest.approx(4.800626903177893e-218, rel=1e-12)


def test_tools_refuse_what_double_precision_cannot_hold():
    cases = [
        ("capital_aggregate", {"standalone": [1e308, 1e308], "correlation": [[1, 1], [1, 1]]}, "RP304", "standalone"),
        ("copula_dependence", {"family": "t", "rho": 0.9, "df": 0.01}, "RP304", "df"),
        ("capital_allocation", {"means": [1e300, 1e300], "sds": [1e300, 1e300], "correlation": [[1, 0], [0, 1]],
                                "seed": 1, "trials": 1000}, "RP304", "sds[0]"),
        ("sensitivity", {"tool": "black_scholes", "parameter": "S", "bump": 1e-300,
                         "arguments": {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}}, "RP102", "bump"),
    ]
    for name, args, code, param in cases:
        err = tools.call(name, args, errors="return")["error"]
        assert (err["code"], err["param"]) == (code, param), (name, err)
    # large but representable figures are scaled, not refused
    big = tools.call("capital_aggregate", {"standalone": [3e200, 4e200], "correlation": [[1, 0], [0, 1]]})
    assert big["total"] == pytest.approx(5e200, rel=1e-14) and big["contributions"]["risk_1"] == pytest.approx(1.8e200)
    # a t copula with almost no degrees of freedom still simulates, without warnings or infinities
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        u = StudentTCopula(0.3, df=1e-3).sample(2000, seed=1)
    assert np.all((u > 0.0) & (u < 1.0))
    # a NaN in a scenario comes back as that scenario's coded error, in strict JSON
    import json

    out = tools.call("scenario_table", {"tool": "black_scholes", "scenarios": {"bad": {"S": float("nan")}},
                                        "arguments": {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}})
    assert out["scenarios"]["bad"]["error"]["code"] == "RP101" and out["scenarios"]["bad"]["overrides"] == {"S": None}
    json.dumps(out, allow_nan=False)


def test_allocations_plot_through_the_viz_layer():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    alloc = allocate({"a": np.arange(100.0), "b": np.arange(100.0)[::-1] * 0.5}, level=0.9)
    fig = alloc.plot()
    assert isinstance(fig, Figure) and "TVaR 0.9 by line (euler)" in fig.axes[0].get_title(loc="left")
    assert isinstance(aggregate_capital([3.0, 4.0], [[1, 0], [0, 1]]).plot(title="Capital"), Figure)
