import pytest
from riskpy import ExperienceRating, ExposureRating, RateAnalyzer

def test_experience_rating_bounds():
    with pytest.raises(Exception):
        ExperienceRating(1082.0, 0.1).experience_mod_factor(-500.0, 1000.0)
    with pytest.raises(Exception):
        ExperienceRating(1082.0, 0.1).experience_mod_factor(500.0, -1000.0)
    
    assert ExperienceRating(1082.0, 0.1).experience_mod_factor(1000.0, 1000.0) == 1.0

def test_exposure_rating_bounds():
    with pytest.raises(Exception):
        ExposureRating(-1000.0, 1.5).increased_limits_factor(5000.0)
    with pytest.raises(Exception):
        ExposureRating(1000.0, 1.5).layer_premium(-500.0, 1000.0, 1.0)
    with pytest.raises(Exception):
        ExposureRating(1000.0, 1.5).burning_cost([500.0, -100.0], 100.0, 1000.0, 5)

def test_rate_analyzer_bounds():
    with pytest.raises(Exception):
        RateAnalyzer(1.0, 0.3).rate_change_impact(-500.0, 600.0)
    with pytest.raises(Exception):
        RateAnalyzer(-1.0, 0.3)
    
    # Check out of bounds vector size mismatch
    with pytest.raises(Exception):
        RateAnalyzer(1.0, 0.3).on_level_premiums([100.0, 200.0], [0.05, 0.06, 0.02])
