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

def test_layer_premium_from_zero_attachment():
    # Regression: attachment=0 threw, because the power-curve ILF is only defined
    # for limits > 0. ILF(0) is 0, so a ground-up layer is p * ILF(limit) / ILF(base).
    er = ExposureRating(100000.0, 0.5)
    ground_up = er.layer_premium(1000.0, 0.0, 250000.0)
    assert ground_up == pytest.approx(1000.0 * er.increased_limits_factor(250000.0) / er.increased_limits_factor(100000.0))

    # Layers stack, and a tiny positive attachment gives the same answer.
    assert er.layer_premium(1000.0, 0.0, 100000.0) + er.layer_premium(1000.0, 100000.0, 150000.0) == pytest.approx(ground_up)
    assert er.layer_premium(1000.0, 1e-9, 250000.0) == pytest.approx(ground_up, rel=1e-6)

    # b = 0 makes the ILF linear: the [0, 2] layer on a base of 1 is twice the premium.
    assert ExposureRating(1.0, 0.0).layer_premium(1.0, 0.0, 2.0) == pytest.approx(2.0)

    with pytest.raises(Exception):
        er.layer_premium(1000.0, -1.0, 250000.0)

def test_rate_analyzer_bounds():
    with pytest.raises(Exception):
        RateAnalyzer(1.0, 0.3).rate_change_impact(-500.0, 600.0)
    with pytest.raises(Exception):
        RateAnalyzer(-1.0, 0.3)
    
    # Check out of bounds vector size mismatch
    with pytest.raises(Exception):
        RateAnalyzer(1.0, 0.3).on_level_premiums([100.0, 200.0], [0.05, 0.06, 0.02])
