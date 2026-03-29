import pytest
from riskpy import MonteCarloSimulator

def test_aggregate_loss_bounds():
    with pytest.raises(Exception):
        MonteCarloSimulator.simulate_aggregate_loss(-10, 5.0, 10.0, 1.5)
    with pytest.raises(Exception):
        MonteCarloSimulator.simulate_aggregate_loss(1000, -5.0, 10.0, 1.5)
    with pytest.raises(Exception):
        MonteCarloSimulator.simulate_aggregate_loss(1000, 5.0, 10.0, -1.5)

def test_aggregate_loss_execution():
    res = MonteCarloSimulator.simulate_aggregate_loss(100, 5.0, 10.0, 1.5)
    assert len(res) == 100

def test_life_portfolio_bounds():
    with pytest.raises(Exception):
        MonteCarloSimulator.simulate_life_portfolio(-100, 1000, 0.05, 0.1, 50000.0)
    with pytest.raises(Exception):
        MonteCarloSimulator.simulate_life_portfolio(100, -1000, 0.05, 0.1, 50000.0)
    with pytest.raises(Exception):
        MonteCarloSimulator.simulate_life_portfolio(100, 1000, 1.5, 0.1, 50000.0)
