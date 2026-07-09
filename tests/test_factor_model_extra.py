import pytest
from riskpy import FactorModel


def test_rule_stacking_same_field():
    model = FactorModel(100.0)
    # Two exact rules that both match → multiplicative stack
    model.add_multiplier("tier", "gold", 1.2)
    model.add_multiplier("tier", "gold", 1.1)
    assert model.calculate({"tier": "gold"}) == pytest.approx(100.0 * 1.2 * 1.1)


def test_inclusive_band_boundaries():
    model = FactorModel(100.0)
    model.add_numeric_band_multiplier("age", 18.0, 25.0, 1.5)
    assert model.calculate({"age": 18.0}) == pytest.approx(150.0)
    assert model.calculate({"age": 25.0}) == pytest.approx(150.0)
    assert model.calculate({"age": 17.9}) == pytest.approx(100.0)
    assert model.calculate({"age": 25.1}) == pytest.approx(100.0)


def test_type_mismatch_skips_rule():
    model = FactorModel(100.0)
    model.add_multiplier("state", "FL", 2.0)
    model.add_numeric_band_multiplier("age", 16.0, 25.0, 1.5)
    # Wrong types: string field gets double, numeric field gets string
    assert model.calculate({"state": 12.0, "age": "young"}) == pytest.approx(100.0)


def test_empty_inputs_returns_base():
    model = FactorModel(250.0)
    model.add_multiplier("class", "A", 1.5)
    assert model.calculate({}) == pytest.approx(250.0)


def test_multi_field_combined():
    model = FactorModel(1000.0)
    model.add_multiplier("state", "FL", 3.0)
    model.add_multiplier("state", "CA", 2.0)
    model.add_numeric_band_multiplier("age", 16.0, 25.0, 2.0)
    model.add_numeric_band_multiplier("age", 26.0, 65.0, 1.0)
    assert model.calculate({"state": "FL", "age": 19.0}) == pytest.approx(6000.0)
    assert model.calculate({"state": "CA", "age": 40.0}) == pytest.approx(2000.0)


def test_set_base_rate():
    model = FactorModel(100.0)
    model.set_base_rate(200.0)
    model.add_multiplier("x", "y", 1.5)
    assert model.calculate({"x": "y"}) == pytest.approx(300.0)


def test_unmatched_fields_ignored():
    model = FactorModel(50.0)
    model.add_multiplier("a", "hit", 2.0)
    assert model.calculate({"a": "miss", "b": "whatever"}) == pytest.approx(50.0)


def test_validation_rejects_bad_rules():
    with pytest.raises(Exception):
        FactorModel(float("nan"))
    model = FactorModel(100.0)
    with pytest.raises(Exception):
        model.add_multiplier("", "x", 1.0)
    with pytest.raises(Exception):
        model.add_multiplier("f", "x", float("inf"))
    with pytest.raises(Exception):
        model.add_numeric_band_multiplier("age", 30.0, 10.0, 1.2)  # min > max
    with pytest.raises(Exception):
        model.set_base_rate(float("nan"))
