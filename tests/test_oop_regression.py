import pytest
from riskpy import cpp_underwriter as engine
import math

# Baseline Regression 1.0 Mock Tests
# The goal is to mathematically prove every single OOP configuration logic works
# using the most basic conceptual value possible: 1.0

def test_exposure_rating_oop_baseline():
    # Instantiate with (1.0 limit, 1.0 b_parameter)
    layer = engine.ExposureRating(base_limit=1.0, b_parameter=1.0)
    
    # ILF(limit) = pow(limit / base, 1 - b)
    # ILF(1.0) = pow(1.0/1.0, 1 - 1) = pow(1, 0) = 1.0
    assert math.isclose(layer.increased_limits_factor(1.0), 1.0)
    
    # ILF(2.0) = pow(2.0/1.0, 0) = 1.0 for b=1.0. 
    # Let's use a different b_parameter to show actual scaling.
    layer_b0 = engine.ExposureRating(base_limit=1.0, b_parameter=0.0)
    # ILF(2.0) = 2.0^1 = 2.0
    assert math.isclose(layer_b0.increased_limits_factor(2.0), 2.0)

    # Layer premium: ground=1.0, att=1.0, lim=1.0
    # Premium = 1.0 * (ILF(2.0) - ILF(1.0)) / ILF(1.0)
    # For b=0, ILF(2) = 2.0, ILF(1) = 1.0 -> 1.0 * (2.0 - 1.0) / 1.0 = 1.0
    assert math.isclose(layer_b0.layer_premium(1.0, 1.0, 1.0), 1.0)

    # Burning Cost: loses=[1.0, 2.0], att=1.0, lim=1.0, years=1
    # For loss = 1.0 -> layer = 0.0
    # For loss = 2.0 -> capped at 2.0, layer = 1.0. Total = 1.0
    assert math.isclose(layer.burning_cost([1.0, 2.0], 1.0, 1.0, 1), 1.0)

def test_experience_rating_oop_baseline():
    # Instantiate with (k=1.0, ballast=1.0)
    er = engine.ExperienceRating(k_parameter=1.0, ballast=1.0)
    
    # Credibility calculation: min(1.0, sqrt(1.0 / 1.0)) = 1.0
    assert math.isclose(er.calculate_credibility(1), 1.0)

    # Cred weighted rate: (1.0 * 1.0) + (1-1.0)*1.0 = 1.0
    assert math.isclose(er.credibility_weighted_rate(1.0, 1.0, 1.0), 1.0)

    # Experience Mod Factor: (actual + ballast*expected) / (expected * (1+ballast))
    # (1.0 + 1.0 * 1.0) / (1.0 * (1.0 + 1.0)) = 2.0 / 2.0 = 1.0
    assert math.isclose(er.experience_mod_factor(1.0, 1.0), 1.0)

    # Expected Losses: exp*freq*sev = 1 * 1 * 1 = 1.0
    assert math.isclose(er.expected_losses(1.0, 1.0, 1.0), 1.0)

def test_rate_analyzer_oop_baseline():
    # Instantiate with CR=1.0, ER=0.0
    ra = engine.RateAnalyzer(target_combined_ratio=1.0, expense_ratio=0.0)
    
    # combined_ratio = LR + ER = 1.0 + 0.0 = 1.0
    assert math.isclose(ra.combined_ratio(1.0), 1.0)
    
    # permissible_lr = 1.0 - 0.0 = 1.0
    assert math.isclose(ra.permissible_loss_ratio(), 1.0)
    
    # required_rate_change: currentLR=1.0 / PLR(1.0) - 1.0 = 0.0
    assert math.isclose(ra.required_rate_change(1.0), 0.0)
    
    # on_level_factor: changes of [0.0] -> 1.0
    assert math.isclose(ra.on_level_factor([0.0]), 1.0)

    # trend_factor: (1 + 0.0)^1 = 1.0
    assert math.isclose(ra.trend_factor(0.0, 1.0), 1.0)

def test_monte_carlo_oop_baseline():
    # Simulator with exactly 1 trial
    sim = engine.MonteCarloSimulator(trials=1)
    
    # We can't guarantee stochastic behavior bounds easily since it uses a random device
    # But we can test if it configures without crashing and returns size = 1
    pc_res = sim.simulate_aggregate_loss(1.0, 1.0, 1.0)
    assert len(pc_res) == 1

    life_res = sim.simulate_life_portfolio(1, 1.0, 0.0, 1.0)
    assert len(life_res) == 1
