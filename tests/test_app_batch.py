import pytest
from riskpy import UnderwritingApp
import os
import csv

def test_batch_calculation_and_export():
    app = UnderwritingApp(title="Test", excel_template="corporate_layout.xlsx")
    
    # We must configure fields manually before calling batch so exporter knows what to map
    app.add_field("age", "Client Age", "A")
    app.add_field("property_value", "Property Value ($)", "B")
    app.set_premium_column("C", "Final Premium")
    
    def mock_logic(inputs):
        return inputs.get("property_value", 0.0) * 0.01

    app.set_logic(mock_logic)

    csv_path = "test_batch_input.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["age", "property_value"])
        writer.writeheader()
        writer.writerow({"age": 30, "property_value": 500000})
        writer.writerow({"age": 45, "property_value": 750000})
        writer.writerow({"age": 60, "property_value": 1500000})

    out_file = "test_batch_quotes.xlsx"
    total_prem, success_count = app.calculate_batch(csv_path, out_file)

    assert success_count == 3
    # 5K + 7.5K + 15K = 27.5K
    assert total_prem == 27500.0

    assert os.path.exists(out_file)

    os.remove(csv_path)
    os.remove(out_file)
