import csv

import pytest

try:
    from riskpy import FactorModel, UnderwritingApp
except ImportError as exc:
    pytest.skip(
        f"UnderwritingApp unavailable (headless / no Tkinter): {exc}",
        allow_module_level=True,
    )


def test_batch_calculation_and_export(tmp_path):
    app = UnderwritingApp(title="Test")

    # We must configure fields manually before calling batch so exporter knows what to map
    app.add_field("age", "Client Age", "A")
    app.add_field("property_value", "Property Value ($)", "B")
    app.set_premium_column("C", "Final Premium")

    def mock_logic(inputs):
        return inputs.get("property_value", 0.0) * 0.01

    app.set_logic(mock_logic)

    csv_path = tmp_path / "batch_input.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["age", "property_value"])
        writer.writeheader()
        writer.writerow({"age": 30, "property_value": 500000})
        writer.writerow({"age": 45, "property_value": 750000})
        writer.writerow({"age": 60, "property_value": 1500000})

    out_file = tmp_path / "batch_quotes.xlsx"
    total_prem, success_count = app.calculate_batch(str(csv_path), str(out_file))

    assert success_count == 3
    # 5K + 7.5K + 15K = 27.5K
    assert total_prem == 27500.0
    assert out_file.exists()


def test_python_logic_headless_quote_and_export(tmp_path):
    # Formerly the root-level test_headless.py script.
    app = UnderwritingApp(title="Test")
    app.add_field("age", "Client Age", "A")
    app.add_field("property_value", "Property Value ($)", "B")
    app.add_field("coverage_amount", "Coverage Amount ($)", "C")
    app.set_premium_column("D", "Final Premium")

    def calculate_premium(inputs):
        base = 500
        if inputs.get("age", 0) < 25:
            base += 200
        if inputs.get("property_value", 0) > 500000:
            base += 300
        if inputs.get("coverage_amount", 0) > 1000000:
            base += 500
        return base

    app.set_logic(calculate_premium)
    quote = {"age": 24.0, "property_value": 600000.0, "coverage_amount": 1500000.0}
    assert app.calculate_headless(quote) == 1500.0

    out_file = tmp_path / "policy_quote.xlsx"
    app.export_excel_headless(str(out_file))
    assert out_file.exists()


def test_factor_model_drives_quotes_and_batches(tmp_path):
    # Formerly the root-level test_batch.py script, now with exact expected values.
    model = FactorModel(initial_base_rate=500.0)
    model.add_multiplier("state", "FL", 3.0)
    model.add_numeric_band_multiplier("age", 18, 25, 1.4)

    app = UnderwritingApp(title="Test")
    app.add_field("state", "Location State", "A", choices=["NY", "FL"])
    app.add_field("age", "Client Age", "B")
    app.set_premium_column("C", "Final Premium")
    app.set_factor_model(model)

    assert app.calculate_headless({"state": "FL", "age": 22.0}) == pytest.approx(2100.0)
    assert app.calculate_headless({"state": "NY", "age": 40.0}) == 500.0

    csv_path = tmp_path / "book.csv"
    csv_path.write_text("state,age\nFL,22\nNY,40\nFL,30\n")
    out_file = tmp_path / "book.xlsx"
    total, count = app.calculate_batch(str(csv_path), str(out_file))

    assert count == 3
    assert total == pytest.approx(2100.0 + 500.0 + 1500.0)
    assert out_file.exists()
