import pytest
from riskpy import FactorModel

def test_factor_model_exact_match():
    model = FactorModel(100.0)
    model.add_multiplier("class", "sedan", 1.2)
    model.add_multiplier("class", "suv", 1.5)
    
    # Hit Sedan rule
    premium1 = model.calculate({"class": "sedan"})
    assert premium1 == 120.0
    
    # Hit SUV rule
    premium2 = model.calculate({"class": "suv"})
    assert premium2 == 150.0
    
    # Non-hit maintains base
    premium3 = model.calculate({"class": "truck"})
    assert premium3 == 100.0

def test_factor_model_numeric_bands():
    model = FactorModel(100.0)
    # Band rule adds 1.5 if age is 18 to 25
    model.add_numeric_band_multiplier("age", 18.0, 25.0, 1.5)
    
    assert model.calculate({"age": 20.0}) == 150.0
    assert model.calculate({"age": 30.0}) == 100.0

def test_factor_model_combined():
    model = FactorModel(100.0)
    model.add_multiplier("city", "NY", 2.0)
    model.add_numeric_band_multiplier("age", 18.0, 25.0, 1.5)
    
    # Both rules apply -> 100 * 2.0 * 1.5 = 300
    assert model.calculate({"city": "NY", "age": 20.0}) == 300.0
    
    # One rule applies
    assert model.calculate({"city": "LA", "age": 20.0}) == 150.0
