"""0.8.0 refinements to riskpy.mc: Model.stress, reverse_stress, LHS under a copula, validation.

Each test pins a defect found by the 0.7.0 stress test. The oracles are exact where the
fix makes them exact — a Distribution shock under a copula must reproduce, value for value,
the column a re-run with the new marginal draws — and statistical otherwise, with
tolerances of several standard errors.
"""
import math

import pytest

np = pytest.importorskip("numpy")

from riskpy import mc  # noqa: E402
from riskpy.capital import GaussianCopula, GumbelCopula, StudentTCopula  # noqa: E402
from riskpy.errors import ArgumentTypeError, InputError, RiskPyError  # noqa: E402
from riskpy.mc import (  # noqa: E402
    Bernoulli, Binomial, Compound, Constant, Exponential, LogNormal, Model, Normal, Poisson, Result, Uniform,
    iman_conover, simulate,
)


def _code(make):
    with pytest.raises(RiskPyError) as exc:
        make()
    return exc.value


def _spearman(x, y):
    rx, ry = mc._rankdata(np.asarray(x), np), mc._rankdata(np.asarray(y), np)
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Model.stress: a Distribution shock on a variable with ties


@pytest.mark.parametrize("base, shock", [
    (Bernoulli(0.1), Bernoulli(0.2)),           # a default flag
    (Poisson(2.0), Poisson(3.5)),               # a claim count
    (Binomial(10, 0.1), Binomial(10, 0.3)),
])
def test_a_discrete_shock_under_a_copula_is_the_same_copula_re_run(base, shock):
    # Under a copula the stressed column is shock.ppf at the copula's own uniforms: exactly
    # what Model.run draws for that variable with the new marginal, the same copula and seed.
    cop = GumbelCopula(3.0)
    fn = lambda d, x: d * x  # noqa: E731
    model = Model(d=base, x=LogNormal(0.0, 1.0))
    stress = model.stress({"s": {"d": shock}}, trials=50_000, seed=4, fn=fn, copula=cop)
    rerun = Model(d=shock, x=LogNormal(0.0, 1.0)).run(50_000, seed=4, fn=fn, copula=cop)
    assert np.array_equal(stress.scenarios["s"].inputs["d"], rerun.inputs["d"])
    assert np.array_equal(stress.scenarios["s"].values, rerun.values)
    # and the base is Model.run on the same seed, trial for trial
    assert np.array_equal(stress.base.values, model.run(50_000, seed=4, fn=fn, copula=cop).values)


def test_the_default_flag_case_from_the_report():
    # d = Bernoulli(10%) -> 20% under Gumbel(3): E[d'x] was 0.72 (ties broken by trial number,
    # so the extra defaults ignored x) against 0.878 on the same latent draws.
    stats = pytest.importorskip("scipy.stats")
    n, cop = 200_000, GumbelCopula(3.0)
    u = cop.uniforms(n, np.random.default_rng(4))
    exact = float(np.mean(np.exp(stats.norm.ppf(u[:, 1])) * (u[:, 0] > 0.8)))
    stress = Model(d=Bernoulli(0.1), x=LogNormal(0, 1)).stress(
        {"s": {"d": Bernoulli(0.2)}}, trials=n, seed=4, fn=lambda d, x: d * x, copula=cop)
    assert stress.scenarios["s"].mean == pytest.approx(exact, rel=1e-9)


def test_independent_tied_variables_stay_independent_when_stressed():
    # 20 independent obligors, PD 2% -> 5%. Breaking every obligor's ties by the same trial
    # order put all the extra defaults on the same last trials: VaR 99.5% of 20 defaults
    # against 4 on a re-run, and thousands of trials in which everyone defaulted.
    names = [f"o{i}" for i in range(20)]
    fn = lambda **kw: sum(kw.values())  # noqa: E731
    n = 200_000
    stress = Model(**{k: Bernoulli(0.02) for k in names}).stress(
        {"pd 5%": {k: Bernoulli(0.05) for k in names}}, trials=n, seed=3, fn=fn)
    s = stress.scenarios["pd 5%"]
    rerun = Model(**{k: Bernoulli(0.05) for k in names}).run(n, seed=9, fn=fn)
    assert abs(s.var(0.995) - rerun.var(0.995)) <= 1.0
    # the stressed portfolio is Binomial(20, 5%): P(L = 20) = 1e-26, mean 1, variance 0.95
    assert int((s.values >= 10).sum()) <= 2
    assert s.mean == pytest.approx(1.0, abs=5 * math.sqrt(0.95 / n))
    assert s.std ** 2 == pytest.approx(0.95, rel=0.03)
    flags = np.column_stack([s.inputs[k] for k in names])
    off = np.corrcoef(flags, rowvar=False)[~np.eye(20, dtype=bool)]
    assert np.max(np.abs(off)) < 0.02


def test_ties_under_a_correlation_keep_the_correlation():
    # Model.correlate (Iman-Conover): ties are broken by the variable's normal score, so the
    # redrawn flags keep the rank dependence a re-run with the new marginal has.
    n = 200_000
    m = Model(d=Bernoulli(0.1), x=LogNormal(0, 1)).correlate("d", "x", 0.6)
    s = m.stress({"s": {"d": Bernoulli(0.2)}}, trials=n, seed=4, fn=lambda d, x: d * x).scenarios["s"]
    reruns = [Model(d=Bernoulli(0.2), x=LogNormal(0, 1)).correlate("d", "x", 0.6).run(n, seed=40 + i,
              fn=lambda d, x: d * x) for i in range(3)]
    target = np.mean([_spearman(r.inputs["d"], r.inputs["x"]) for r in reruns])
    assert _spearman(s.inputs["d"], s.inputs["x"]) == pytest.approx(target, abs=0.01)
    mean = np.mean([r.mean for r in reruns])
    se = max(r.std for r in reruns) / math.sqrt(n)
    assert s.mean == pytest.approx(mean, abs=6 * se)   # was 0.554 against 0.677


def test_a_zero_inflated_compound_shock_under_a_copula_keeps_the_dependence():
    # A cat line with most years claim-free, frequency doubled. A Compound has no quantile
    # function, so a fresh sample is sorted into the copula's ranks — ties (the zero years)
    # by the copula's uniform, not by trial number.
    n, cop = 200_000, GaussianCopula(0.7)
    base = dict(cat=Compound(Poisson(0.3), LogNormal(3, 1)), motor=LogNormal(4, 0.3))
    new = Compound(Poisson(0.6), LogNormal(3, 1))
    fn = lambda cat, motor: cat + motor  # noqa: E731
    s = Model(**base).stress({"s": {"cat": new}}, trials=n, seed=4, fn=fn, copula=cop).scenarios["s"]
    rerun = Model(**{**base, "cat": new}).run(n, seed=4, fn=fn, copula=cop)
    assert _spearman(s.inputs["cat"], s.inputs["motor"]) == pytest.approx(
        _spearman(rerun.inputs["cat"], rerun.inputs["motor"]), abs=0.01)    # was 0.33 against 0.47 (tau)
    cross = lambda r: float(np.mean(r.inputs["cat"] * r.inputs["motor"]))  # noqa: E731
    assert cross(s) == pytest.approx(cross(rerun), rel=0.03)
    assert s.mean == pytest.approx(rerun.mean, abs=5 * math.hypot(s.std, rerun.std) / math.sqrt(n))


def test_the_base_of_a_stress_is_bit_identical_to_run():
    fn = lambda a, b: a + b  # noqa: E731
    dists = dict(a=Poisson(3.0), b=LogNormal(1.0, 0.5))
    for kwargs in ({}, {"copula": StudentTCopula(0.4, df=4)}, {"sampling": "lhs"},
                   {"copula": GumbelCopula(2.0), "sampling": "lhs"}):
        m = Model(**dists)
        stress = m.stress({"s": {"a": Poisson(5.0)}}, trials=5_000, seed=12, fn=fn, **kwargs)
        assert np.array_equal(stress.base.values, m.run(5_000, seed=12, fn=fn, **kwargs).values), kwargs
    m = Model(**dists).correlate("a", "b", 0.5)
    stress = m.stress({"s": {"a": Poisson(5.0)}}, trials=5_000, seed=12, fn=fn)
    assert np.array_equal(stress.base.values, m.run(5_000, seed=12, fn=fn).values)


# ---------------------------------------------------------------------------
# Model.stress: a callable shock, scenario order, NumPy scalars


def test_a_callable_that_changes_its_array_in_place_cannot_leak():
    def doubled(x):
        x *= 2.0
        return x

    model = Model(a=Normal(10.0, 1.0), b=Normal(0.0, 1.0))
    fn = lambda a, b: a + b  # noqa: E731
    reference = model.run(2_000, seed=1, fn=fn)
    stress = model.stress({"double": {"a": doubled}, "pin b": {"b": 1.0}}, trials=2_000, seed=1, fn=fn)
    assert np.array_equal(stress.base.values, reference.values)
    assert np.array_equal(stress.base.inputs["a"], reference.inputs["a"])
    assert np.allclose(stress.scenarios["double"].inputs["a"], 2.0 * reference.inputs["a"], rtol=1e-15)
    # a later scenario sees the base draws, not the doubled ones
    assert np.array_equal(stress.scenarios["pin b"].values, reference.inputs["a"] + 1.0)


def test_scenario_results_do_not_depend_on_the_other_scenarios():
    m = Model(claims=Compound(Poisson(mean=120), LogNormal(mu=8.5, sigma=1.2)), inflation=Normal(0.04, 0.01))
    m.correlate("claims", "inflation", 0.3)
    fn = lambda claims, inflation: claims * (1 + inflation)  # noqa: E731
    heavier = {"claims": Compound(Poisson(120), LogNormal(8.5, 1.4))}
    other = {"inflation": Normal(0.05, 0.02)}
    alone = m.stress({"heavier tail": heavier}, trials=20_000, seed=1, fn=fn)
    after = m.stress({"inflation shock": other, "heavier tail": heavier}, trials=20_000, seed=1, fn=fn)
    before = m.stress({"heavier tail": heavier, "inflation shock": other}, trials=20_000, seed=1, fn=fn)
    for s in (after, before):
        assert np.array_equal(s.scenarios["heavier tail"].values, alone.scenarios["heavier tail"].values)
    assert np.array_equal(after.scenarios["inflation shock"].values, before.scenarios["inflation shock"].values)
    # the same redraw under another label, or in two scenarios at once, is the same numbers
    renamed = m.stress({"x": heavier, "y": {**heavier, **other}}, trials=20_000, seed=1, fn=fn)
    assert np.array_equal(renamed.scenarios["x"].values, alone.scenarios["heavier tail"].values)
    assert np.array_equal(renamed.scenarios["y"].inputs["claims"], renamed.scenarios["x"].inputs["claims"])
    # a stress with no Distribution shock draws nothing after the base: a generator passed as
    # the seed is left exactly where the base run left it
    gen_a, gen_b = np.random.default_rng(5), np.random.default_rng(5)
    m.stress({"p": {"inflation": 0.08}}, trials=1_000, seed=gen_a, fn=fn)
    m.run(1_000, seed=gen_b, fn=fn)
    assert gen_a.random() == gen_b.random()


def test_numpy_scalars_pin_a_variable_and_booleans_do_not():
    model = Model(a=Normal(0.0, 1.0), b=Normal(0.0, 1.0))
    fn = lambda a, b: a + b  # noqa: E731
    for value in (np.int64(1), np.int32(1), np.float32(1.0), np.float64(1.0), 1):
        stress = model.stress({"s": {"a": value}}, trials=100, seed=1, fn=fn)
        assert np.all(stress.scenarios["s"].inputs["a"] == 1.0), type(value)
    for value in (True, np.bool_(True), "1", None):
        err = _code(lambda: model.stress({"s": {"a": value}}, trials=100, seed=1, fn=fn))
        assert err.code == "RP404" and err.param == "scenarios", value
    assert _code(lambda: model.stress({"s": {"a": np.float64("nan")}}, trials=100, fn=fn)).code == "RP101"


def test_a_bad_shock_is_refused_before_the_base_run():
    calls = []

    def fn(a):
        calls.append(1)
        return a

    with pytest.raises(ArgumentTypeError):
        Model(a=Normal(0, 1)).stress({"ok": {"a": 1.0}, "bad": {"a": "x"}}, trials=10, fn=fn)
    assert calls == []


# ---------------------------------------------------------------------------
# Latin hypercube sampling under a copula


def test_lhs_under_a_copula_stratifies_and_keeps_the_copula_ranks():
    n = 20_000
    model = Model(a=Uniform(0.0, 1.0), b=Exponential(1.0), c=Compound(Poisson(2.0), LogNormal(0.0, 0.5)))
    cop = GaussianCopula(np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]]))
    rand = model.sample(n, seed=5, copula=cop)
    lhs = model.sample(n, seed=5, copula=cop, sampling="lhs")
    # exactly one draw in each 1/n stratum of every ppf-able marginal (was: 37% of strata empty)
    assert np.array_equal(np.sort(np.floor(lhs["a"] * n)), np.arange(n))
    assert np.array_equal(np.sort(np.floor(-np.expm1(-lhs["b"]) * n)), np.arange(n))
    # the dependence is the copula's: the same uniforms, so the same ranks as a random draw
    for name in ("a", "b"):
        assert np.array_equal(mc._ranks(lhs[name]), mc._ranks(rand[name]))
    assert _spearman(lhs["c"], lhs["a"]) == pytest.approx(_spearman(rand["c"], rand["a"]), abs=0.02)


def test_lhs_under_a_copula_cuts_the_noise():
    fn = lambda a, b: a + b  # noqa: E731

    def spread(sampling):
        return float(np.std([simulate(fn, trials=5_000, seed=s, sampling=sampling, copula=GaussianCopula(0.5),
                                      a=LogNormal(4.0, 0.5), b=mc.Gamma(2.0, 30.0)).mean for s in range(20)], ddof=1))

    # sampling="lhs" used to be ignored under a copula: the two spreads were identical
    assert spread("random") / spread("lhs") > 10.0


# ---------------------------------------------------------------------------
# Correlation matrices


def test_a_correlation_matrix_with_infinite_or_nan_entries_is_refused():
    inf, nan = float("inf"), float("nan")
    for matrix in ([[1.0, inf], [inf, 1.0]], [[1.0, -inf], [-inf, 1.0]]):
        for make in (lambda: GaussianCopula(matrix), lambda: StudentTCopula(matrix, df=4),
                     lambda: simulate(lambda a, b: a + b, trials=10, correlation=matrix, a=Normal(0, 1), b=Normal(0, 1)),
                     lambda: iman_conover({"a": np.zeros(5), "b": np.zeros(5)}, matrix, np.random.default_rng(0))):
            err = _code(make)
            assert err.code == "RP101" and err.param == "correlation" and "finite" in err.message
    err = _code(lambda: GaussianCopula([[1.0, nan], [nan, 1.0]]))
    assert err.code == "RP101" and "NaN" in err.message and "symmetric" not in err.message
    err = _code(lambda: simulate(lambda a, b: a, trials=10, correlation=[[1.0, 1.5], [1.5, 1.0]],
                                 a=Normal(0, 1), b=Normal(0, 1)))
    assert err.code == "RP100" and "[-1, 1]" in err.message
    err = _code(lambda: simulate(lambda a, b: a, trials=10, correlation=[[1.0, 0.5], [0.5]],
                                 a=Normal(0, 1), b=Normal(0, 1)))
    assert err.code == "RP100" and err.param == "correlation"
    err = _code(lambda: simulate(lambda a, b: a, trials=10, correlation=[["1", "x"], ["x", "1"]],
                                 a=Normal(0, 1), b=Normal(0, 1)))
    assert err.code == "RP100" and err.param == "correlation"


def test_a_refused_correlation_is_not_kept_on_the_model():
    m = Model(a=Normal(0, 1), b=Normal(0, 1), c=Normal(0, 1)).correlate("a", "b", 0.9)
    with pytest.raises(InputError, match="positive semi-definite"):
        m.correlate("a", "c", 0.9)               # with b and c independent, impossible
    assert m.correlations == {("a", "b"): 0.9}
    m.run(100, seed=1, fn=lambda a, b, c: a + b + c)      # still runs
    assert _code(lambda: m.correlate("a", "b", "0.3")).code == "RP101"
    assert _code(lambda: m.correlate("a", "b", None)).code == "RP101"


# ---------------------------------------------------------------------------
# Result.convergence


def test_convergence_with_one_two_and_three_trials():
    one = simulate(lambda x: x, 1, seed=1, x=Normal(0, 1))
    n, mean, half = one.convergence()
    assert list(n) == [1] and mean[0] == one.values[0] and half[0] == 0.0
    for trials in (2, 3):
        n, mean, half = simulate(lambda x: x, trials, seed=1, x=Normal(0, 1)).convergence()
        assert n[-1] == trials and np.all(np.isfinite(half))


# ---------------------------------------------------------------------------
# trials and seed


def test_trials_must_be_a_whole_number():
    fn = lambda x: x  # noqa: E731
    model = Model(x=Normal(0, 1))
    model.formula(fn)
    for bad in (1.5, 100.0, True, "100", None, 0, -3):
        for make in (lambda: simulate(fn, bad, x=Normal(0, 1)), lambda: model.run(bad),
                     lambda: model.sample(bad), lambda: model.stress({"s": {"x": 1.0}}, bad)):
            err = _code(make)
            assert err.code == "RP107" and err.param == "trials", (bad, err)
    assert simulate(fn, np.int64(50), seed=1, x=Normal(0, 1)).trials == 50
    assert type(model.run(np.int32(20), seed=1).trials) is int


def test_seed_must_be_a_non_negative_whole_number():
    fn = lambda x: x  # noqa: E731
    model = Model(x=Normal(0, 1))
    model.formula(fn)
    for bad, code in ((-1, "RP102"), (1.5, "RP107"), (3.0, "RP107"), (True, "RP107"), ("42", "RP107")):
        for make in (lambda: simulate(fn, 10, seed=bad, x=Normal(0, 1)), lambda: model.run(10, seed=bad),
                     lambda: model.sample(10, seed=bad), lambda: model.stress({"s": {"x": 1.0}}, 10, seed=bad)):
            err = _code(make)
            assert err.code == code and err.param == "seed", (bad, err)
    # a NumPy integer is a seed like any other; NumPy's own seed objects still pass through
    assert np.array_equal(model.run(100, seed=np.int64(7)).values, model.run(100, seed=7).values)
    assert np.array_equal(model.run(100, seed=np.random.SeedSequence(7)).values, model.run(100, seed=7).values)
    assert model.run(10, seed=np.random.default_rng(1)).trials == 10
    assert model.run(10, seed=10 ** 30).trials == 10


# ---------------------------------------------------------------------------
# numbers: booleans, strings, None and huge integers


def test_distribution_parameters_refuse_non_numbers_with_a_code():
    for make in (lambda: Normal(True, 1), lambda: Normal("1", 1), lambda: Normal(None, 1), lambda: Normal(0, None),
                 lambda: Normal(10 ** 400, 1), lambda: LogNormal(b"1", 1), lambda: Constant(np.bool_(True)),
                 lambda: Exponential(True), lambda: Uniform(None, 1), lambda: mc.Triangular("0", 1, 2),
                 lambda: Bernoulli(True), lambda: Bernoulli(float("nan")), lambda: Binomial(5, float("nan")),
                 lambda: mc.NegativeBinomial(3, "0.5")):
        err = _code(make)
        assert isinstance(err, InputError) and err.code == "RP101", err


def test_levels_and_percentiles_refuse_non_numbers_with_a_code():
    result = simulate(lambda x: x, 1_000, seed=1, x=Normal(0, 1))
    for make in (lambda: result.var(None), lambda: result.tvar("0.9"), lambda: result.var(True),
                 lambda: result.percentile(None), lambda: result.percentile("50"),
                 lambda: result.reverse_stress(None), lambda: result.reverse_stress(threshold=True),
                 lambda: result.var(10 ** 400)):
        err = _code(make)
        assert err.code == "RP101", err
    assert _code(lambda: result.var(float("nan"))).code == "RP110"
    assert result.var(np.float32(0.99)) == pytest.approx(result.var(0.99), rel=1e-6)
    assert result.percentile(np.int64(50)) == result.median


def test_sd_of_minus_zero_is_zero():
    for dist in (Normal(3.0, -0.0), LogNormal(0.0, -0.0)):
        drawn = simulate(lambda x: x, 100, seed=1, x=dist).values
        assert np.all(drawn == drawn[0])
    assert math.copysign(1.0, Normal(0.0, -0.0).sd) == 1.0


def test_binomial_trials_is_a_whole_number_never_truncated():
    for bad in (2.5, float("nan"), float("inf"), True, "3", None, -1):
        err = _code(lambda: Binomial(bad, 0.3))
        assert err.code == "RP107" and err.param == "trials", bad
    assert Binomial(3.0, 0.3).trials == 3 and Binomial(np.int64(4), 0.5).trials == 4


# ---------------------------------------------------------------------------
# Messages


def test_a_function_passed_to_model_points_at_the_formula_decorator():
    err = _code(lambda: Model(formula=lambda a: a, a=LogNormal(0.0, 1.0)))
    assert err.code == "RP404" and err.param == "formula"
    assert "@model.formula" in err.message and "Constant" not in err.message
    assert "JSON" not in err.hint
    err = _code(lambda: simulate(lambda x: x, 10, x=Normal(0, 1), f=lambda x: x))
    assert err.code == "RP404" and "first argument" in err.message
    # a number still gets the Constant advice, and a class is told to call itself
    err = _code(lambda: Model(rate=0.05))
    assert "Constant(0.05)" in err.message and "Constant(0.05)" in err.hint
    err = _code(lambda: Model(x=Normal))
    assert "Normal(...)" in err.message
    err = _code(lambda: Model(x="normal"))
    assert err.code == "RP404" and "Constant" not in err.message


def test_a_copula_of_the_wrong_dimension_says_how_to_build_one():
    m = Model(a=Normal(0, 1), b=Normal(0, 1), c=Normal(0, 1))
    m.formula(lambda a, b, c: a + b + c)
    err = _code(lambda: m.run(100, seed=1, copula=GumbelCopula(2.0)))
    assert err.code == "RP109" and err.param == "copula"
    assert "dim=3" in err.hint and "3x3" in err.hint and "list" not in err.hint
