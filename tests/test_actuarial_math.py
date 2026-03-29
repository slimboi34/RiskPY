import pytest
from riskpy import ActuarialMath

def test_present_value():
    # Regular case
    assert round(ActuarialMath.present_value(0.05, 10, 1000), 2) == 7721.73
    
    # Zero rate
    assert ActuarialMath.present_value(0.0, 10, 1000) == 10000.0
    
    # Assert invalid rate limits throw Exception
    with pytest.raises(Exception):
        ActuarialMath.present_value(-1.5, 10, 1000)

def test_future_value():
    assert round(ActuarialMath.future_value(0.05, 10, 1000), 2) == 12577.89
    
    with pytest.raises(Exception):
        ActuarialMath.future_value(-1.0, 10, 1000)

def test_loss_ratio():
    assert ActuarialMath.calculate_loss_ratio(500, 1000) == 0.5
    
    with pytest.raises(Exception):
        ActuarialMath.calculate_loss_ratio(-10, 1000)
    
    with pytest.raises(Exception):
        ActuarialMath.calculate_loss_ratio(500, 0)
