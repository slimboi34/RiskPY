import pytest
from riskpy import MonteCarloSimulator

def test_aggregate_loss_bounds():
    with pytest.raises(Exception):
        MonteCarloSimulator(-10).simulate_aggregate_loss(5.0, 10.0, 1.5)
    with pytest.raises(Exception):
        MonteCarloSimulator(1000).simulate_aggregate_loss(-5.0, 10.0, 1.5)
    with pytest.raises(Exception):
        MonteCarloSimulator(1000).simulate_aggregate_loss(5.0, 10.0, -1.5)

def test_aggregate_loss_execution():
    res = MonteCarloSimulator(100).simulate_aggregate_loss(5.0, 10.0, 1.5)
    assert len(res) == 100

def test_life_portfolio_bounds():
    with pytest.raises(Exception):
        MonteCarloSimulator(-100).simulate_life_portfolio(1000, 0.05, 0.1, 50000.0)
    with pytest.raises(Exception):
        MonteCarloSimulator(100).simulate_life_portfolio(-1000, 0.05, 0.1, 50000.0)
    with pytest.raises(Exception):
        MonteCarloSimulator(100).simulate_life_portfolio(1000, 1.5, 0.1, 50000.0)

def test_economic_scenario_bounds():
    with pytest.raises(Exception):
        MonteCarloSimulator(100).simulate_economic_path(-100.0, 0.0, 0.1, 5)
    with pytest.raises(Exception):
        MonteCarloSimulator(100).simulate_economic_path(100.0, 0.0, -0.1, 5)

def test_health_claims_bounds():
    with pytest.raises(Exception):
        MonteCarloSimulator(100).simulate_health_claims(-5.0, 1.0, 1.0, 1.0)
    with pytest.raises(Exception):
        MonteCarloSimulator(100).simulate_health_claims(5.0, -1.0, 1.0, 1.0)
    with pytest.raises(Exception):
        MonteCarloSimulator(100).simulate_health_claims(5.0, 1.0, -1.0, 1.0)

def test_catastrophe_bounds():
    with pytest.raises(Exception):
        MonteCarloSimulator(100).simulate_catastrophe_loss(1.5, 1.0, 1.0)
    with pytest.raises(Exception):
        MonteCarloSimulator(100).simulate_catastrophe_loss(0.5, -1.0, 1.0)


# --- 0.4.1 hardening of the compiled core -----------------------------------------------
# These need the 0.4.1 extension; against an older wheel (the no-toolchain overlay recipe)
# they are skipped, because a NaN frequency there hangs the interpreter rather than failing.
import math as _math

import riskpy as _riskpy


def _version_tuple(v):
    return tuple(int(p) for p in v.split("+")[0].split(".")[:3] if p.isdigit())


needs_041_core = pytest.mark.skipif(_version_tuple(_riskpy.__version__) < (0, 4, 1),
                                    reason="compiled core older than 0.4.1")


@needs_041_core
@pytest.mark.parametrize("call", [
    lambda m: m.simulate_aggregate_loss(_math.nan, 0.0, 1.0),        # used to loop forever
    lambda m: m.simulate_aggregate_loss(1e12, 0.0, 1.0),             # Poisson count overflows int
    lambda m: m.simulate_aggregate_loss(1.0, _math.nan, 1.0),
    lambda m: m.simulate_aggregate_loss(1.0, 0.0, _math.nan),
    lambda m: m.simulate_life_portfolio(10, _math.nan, 0.1, 1.0),
    lambda m: m.simulate_economic_path(_math.nan, 0.05, 0.2, 10),
    lambda m: m.simulate_health_claims(_math.nan, 0.5, 2.0, 100.0),
    lambda m: m.simulate_health_claims(1e12, 0.5, 2.0, 100.0),
    lambda m: m.simulate_catastrophe_loss(_math.nan, 1.0, 2.0),
])
def test_nan_and_oversized_parameters_are_refused(call):
    with pytest.raises(ValueError):
        call(MonteCarloSimulator(trials=3, seed=1))
