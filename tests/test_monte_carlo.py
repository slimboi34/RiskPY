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
