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
                     lambda: model.sample(bad)):
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
                     lambda: model.sample(10, seed=bad)):
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
