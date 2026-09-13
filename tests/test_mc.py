"""Tests for the generic Monte Carlo engine.

Statistical assertions use a fixed seed and generous tolerances. A test that
fails once a month because a random draw was unusual is worse than no test —
people learn to re-run it, and then they re-run it the day it breaks for real.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from riskpy.mc import (  # noqa: E402
    Bernoulli,
    Constant,
    Empirical,
    LogNormal,
    Model,
    NegativeBinomial,
    Normal,
    Pareto,
    PERT,
    Poisson,
    Triangular,
    Uniform,
    simulate,
)

SEED = 12345
N = 60_000


def test_simulate_is_reproducible_with_a_seed():
    kwargs = dict(trials=5_000, seed=7, x=Normal(0.0, 1.0))
    first = simulate(lambda x: x, **kwargs)
    second = simulate(lambda x: x, **kwargs)
    assert np.array_equal(first.values, second.values)


def test_different_seeds_give_different_draws():
    a = simulate(lambda x: x, trials=5_000, seed=1, x=Normal(0.0, 1.0))
    b = simulate(lambda x: x, trials=5_000, seed=2, x=Normal(0.0, 1.0))
    assert not np.array_equal(a.values, b.values)


def test_normal_recovers_its_moments():
    result = simulate(lambda x: x, trials=N, seed=SEED, x=Normal(mean=50.0, sd=8.0))
    assert result.mean == pytest.approx(50.0, abs=0.2)
    assert result.std == pytest.approx(8.0, abs=0.2)


def test_lognormal_from_moments_round_trips():
    """The parameterisation people get wrong most often, so pin it."""
    dist = LogNormal.from_moments(mean=18_000.0, sd=42_000.0)
    result = simulate(lambda s: s, trials=400_000, seed=SEED, s=dist)
    # Heavy right tail, so the sample mean converges slowly — 3% is fair.
    assert result.mean == pytest.approx(18_000.0, rel=0.03)


def test_constant_is_constant():
    result = simulate(lambda c: c, trials=1_000, seed=SEED, c=Constant(3.5))
    assert result.std == 0.0
    assert result.mean == 3.5


def test_poisson_and_negative_binomial_means():
    pois = simulate(lambda n: n, trials=N, seed=SEED, n=Poisson(mean=140.0))
    assert pois.mean == pytest.approx(140.0, rel=0.01)

    nb = NegativeBinomial.from_mean_dispersion(mean=140.0, dispersion=2.5)
    over = simulate(lambda n: n, trials=N, seed=SEED, n=nb)
    assert over.mean == pytest.approx(140.0, rel=0.02)
    # The whole point of the negative binomial: variance exceeds the mean.
    assert over.std ** 2 > over.mean * 1.5


def test_bounded_distributions_stay_in_bounds():
    result = simulate(
        lambda t, p, u: t + p + u,
        trials=20_000,
        seed=SEED,
        t=Triangular(1.0, 2.0, 3.0),
        p=PERT(0.0, 0.25, 1.0),
        u=Uniform(10.0, 20.0),
    )
    assert result.values.min() >= 11.0
    assert result.values.max() <= 24.0


def test_bernoulli_frequency():
    result = simulate(lambda b: b, trials=N, seed=SEED, b=Bernoulli(p=0.3))
    assert result.mean == pytest.approx(0.3, abs=0.01)
    assert set(np.unique(result.values)) <= {0.0, 1.0}


def test_empirical_only_returns_observed_values():
    data = [1.0, 4.0, 9.0]
    result = simulate(lambda e: e, trials=2_000, seed=SEED, e=Empirical(data))
    assert set(np.unique(result.values)) <= set(data)


def test_var_and_tvar_are_ordered():
    result = simulate(
        lambda s: s, trials=N, seed=SEED, s=LogNormal(mu=8.0, sigma=1.1)
    )
    assert result.var(0.95) < result.var(0.995)
    # TVaR averages the losses beyond VaR, so it can never be smaller.
    assert result.tvar(0.95) >= result.var(0.95)
    assert result.tvar(0.995) >= result.var(0.995)


def test_percentile_matches_var():
    result = simulate(lambda x: x, trials=20_000, seed=SEED, x=Normal(0.0, 1.0))
    assert result.var(0.99) == pytest.approx(result.percentile(99.0))


def test_probability_helpers_agree():
    result = simulate(lambda x: x, trials=N, seed=SEED, x=Normal(0.0, 1.0))
    above = result.prob_above(0.0)
    below = result.prob_below(0.0)
    assert above == pytest.approx(0.5, abs=0.01)
    assert above + below == pytest.approx(1.0, abs=1e-9)


def test_standard_error_shrinks_with_root_n():
    small = simulate(lambda x: x, trials=10_000, seed=SEED, x=Normal(0.0, 1.0))
    big = simulate(lambda x: x, trials=90_000, seed=SEED, x=Normal(0.0, 1.0))
    # Nine times the trials should be about three times the precision.
    assert small.standard_error / big.standard_error == pytest.approx(3.0, rel=0.15)


def test_convergence_series_is_well_formed():
    result = simulate(lambda x: x, trials=20_000, seed=SEED, x=Normal(5.0, 2.0))
    n, running, half = result.convergence(points=50)
    assert len(n) == len(running) == len(half)
    assert n[0] >= 1 and n[-1] == 20_000
    # The running mean must end at the overall mean by construction.
    assert running[-1] == pytest.approx(result.mean, rel=1e-9)
    # And the band must tighten.
    assert half[-1] < half[0]


def test_sensitivity_ranks_the_dominant_driver_first():
    result = simulate(
        lambda big, small, flat: 10.0 * big + 0.01 * small + flat,
        trials=30_000,
        seed=SEED,
        big=Normal(0.0, 1.0),
        small=Normal(0.0, 1.0),
        flat=Constant(2.0),
    )
    ranked = result.sensitivity()
    assert ranked[0][0] == "big"
    assert abs(ranked[0][1]) > 0.9
    assert dict(ranked)["flat"] == 0.0


def test_sensitivity_sign_follows_direction():
    result = simulate(
        lambda up, down: up - down,
        trials=20_000,
        seed=SEED,
        up=Normal(0.0, 1.0),
        down=Normal(0.0, 1.0),
    )
    scores = dict(result.sensitivity())
    assert scores["up"] > 0.5
    assert scores["down"] < -0.5


def test_model_decorator_round_trip():
    model = Model(a=Normal(1.0, 0.1), b=Constant(2.0))

    @model.formula
    def product(a, b):
        return a * b

    result = model.run(5_000, seed=SEED)
    assert result.label == "product"
    assert result.mean == pytest.approx(2.0, abs=0.02)


def test_non_vectorised_matches_vectorised():
    def formula(a, b):
        return a * b + 1.0

    kwargs = dict(trials=2_000, seed=SEED, a=Normal(1.0, 0.2), b=Uniform(1.0, 2.0))
    fast = simulate(formula, vectorised=True, **kwargs)
    slow = simulate(formula, vectorised=False, **kwargs)
    assert np.allclose(fast.values, slow.values)


# -- error paths ------------------------------------------------------------


def test_bad_distribution_parameters_fail_at_construction():
    with pytest.raises(ValueError):
        Normal(mean=0.0, sd=-1.0)
    with pytest.raises(ValueError):
        Uniform(low=5.0, high=1.0)
    with pytest.raises(ValueError):
        Triangular(low=0.0, mode=5.0, high=1.0)
    with pytest.raises(ValueError):
        Pareto(xm=1.0, alpha=0.0)


def test_plain_number_is_rejected_with_a_useful_message():
    with pytest.raises(TypeError, match="Constant"):
        Model(x=1.0)


def test_wrong_shape_from_a_formula_is_caught():
    with pytest.raises(ValueError, match="expected"):
        simulate(lambda x: 0.0, trials=100, seed=SEED, x=Normal(0.0, 1.0))


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_non_finite_output_is_caught():
    with pytest.raises(ValueError, match="NaN or infinity"):
        simulate(lambda x: 1.0 / x, trials=1_000, seed=SEED, x=Constant(0.0))


def test_no_variables_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        simulate(lambda: 1.0, trials=10)


def test_missing_formula_is_rejected():
    with pytest.raises(ValueError, match="No formula"):
        Model(x=Normal(0.0, 1.0)).run(10)


def test_bad_confidence_level_is_rejected():
    result = simulate(lambda x: x, trials=1_000, seed=SEED, x=Normal(0.0, 1.0))
    with pytest.raises(ValueError, match="probability"):
        result.var(99.5)


def test_sensitivity_without_inputs_explains_itself():
    result = simulate(
        lambda x: x, trials=1_000, seed=SEED, keep_inputs=False, x=Normal(0.0, 1.0)
    )
    with pytest.raises(ValueError, match="keep_inputs"):
        result.sensitivity()


def test_summary_mentions_the_headline_numbers():
    result = simulate(lambda x: x, trials=5_000, seed=SEED, x=Normal(100.0, 10.0))
    text = result.summary()
    assert "trials" in text and "VaR" in text and "TVaR" in text
    assert "5,000" in text
