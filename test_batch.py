import os
import csv
import random
from riskpy import UnderwritingApp, FactorModel

print("Generating random bulk book of 100 quotes...")
csv_file = "bulk_quotes.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["state", "age", "property_value", "coverage_amount"])
    writer.writeheader()
    for _ in range(100):
        writer.writerow({
            "state": random.choice(["NY", "CA", "TX", "FL"]),
            "age": random.randint(18, 80),
            "property_value": random.randint(100000, 2000000),
            "coverage_amount": random.randint(100000, 5000000)
        })

print("Spinning up UnderwritingApp...")
app = UnderwritingApp(title="Test Tool", excel_template="corporate_layout.xlsx")

# Using C++ compiled FactorModel instead of python callbacks
model = FactorModel(initial_base_rate=500.0)
model.add_multiplier("state", "CA", 2.0)
model.add_multiplier("state", "FL", 3.0)
model.add_multiplier("state", "NY", 1.4)
model.add_numeric_band_multiplier("age", 18, 25, 1.4)
model.add_numeric_band_multiplier("coverage_amount", 1000000, 5000000, 2.0)
model.add_numeric_band_multiplier("property_value", 500000, 10000000, 1.6)

app.set_factor_model(model)

# Field configuration (must match CSV headers)
app.add_field("state", "Location State", "A", choices=["NY", "CA", "TX", "FL"])
app.add_field("age", "Client Age", "B")
app.add_field("property_value", "Property Value ($)", "C")
app.add_field("coverage_amount", "Coverage Amount ($)", "D")
app.set_premium_column("E", "Final Premium")

print("Executing CSV Batch Process...")
total, count = app.calculate_batch("bulk_quotes.csv", "batch_policy_quotes.xlsx")

print(f"Successfully processed {count} records!")
print(f"Total Written Premium calculated: ${total:,.2f}")

assert count == 100, "Failed to process exactly 100 records."
assert total > 50000, "Calculated premium is suspiciously low."
assert os.path.exists("batch_policy_quotes.xlsx"), "Excel bulk payload was not generated."
print("Batch Scale-up testing verified flawlessly!")
