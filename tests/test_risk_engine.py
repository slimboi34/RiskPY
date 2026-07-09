import os

import pytest
from riskpy import ExcelExporter, FactorModel, Field, RiskEngine


def test_execute_without_logic_returns_zero():
    engine = RiskEngine()
    assert engine.execute({"x": 1.0}) == pytest.approx(0.0)


def test_execute_with_python_logic():
    engine = RiskEngine()

    def logic(inputs):
        return float(inputs.get("premium", 0.0)) * 1.1

    engine.set_math_logic(logic)
    assert engine.execute({"premium": 100.0}) == pytest.approx(110.0)


def test_factor_model_as_logic():
    engine = RiskEngine()
    model = FactorModel(100.0)
    model.add_multiplier("state", "TX", 1.25)

    def logic(inputs):
        return model.calculate(inputs)

    engine.set_math_logic(logic)
    assert engine.execute({"state": "TX"}) == pytest.approx(125.0)
    assert engine.execute({"state": "NY"}) == pytest.approx(100.0)


def test_get_fields_returns_added_fields():
    engine = RiskEngine()
    engine.add_field(Field("age", "Age", "number"))
    engine.add_field(Field("state", "State", "string"))
    fields = engine.get_fields()
    assert len(fields) == 2
    assert fields[0].name == "age"
    assert fields[1].name == "state"
    assert fields[0].label == "Age"


def test_get_fields_empty():
    engine = RiskEngine()
    assert list(engine.get_fields()) == []


def test_export_to_excel_with_exporter(tmp_path):
    engine = RiskEngine()
    exporter = ExcelExporter("template.xlsx")
    # map_column(col, field_name, label)
    exporter.map_column("A", "age", "Age")
    exporter.map_column("B", "premium_output", "Premium")
    engine.attach_exporter(exporter)
    engine.set_math_logic(lambda inputs: float(inputs.get("age", 0.0)) * 10.0)
    engine.execute({"age": 30.0})
    out = str(tmp_path / "quote.xlsx")
    engine.export_to_excel(out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_export_batch_to_excel(tmp_path):
    engine = RiskEngine()
    exporter = ExcelExporter("template.xlsx")
    exporter.map_column("A", "age", "Age")
    exporter.map_column("B", "premium_output", "Premium")
    engine.attach_exporter(exporter)
    batch_inputs = [{"age": 20.0}, {"age": 40.0}]
    batch_prem = [200.0, 400.0]
    out = str(tmp_path / "batch.xlsx")
    engine.export_batch_to_excel(out, batch_inputs, batch_prem)
    assert os.path.exists(out)
