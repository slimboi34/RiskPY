# headless test for the C++ Underwriting Framework using Wrapper
from riskpy import UnderwritingApp
import os

print("Spinning up UnderwritingApp...")
app = UnderwritingApp(title="Test Tool", excel_template="corporate_layout.xlsx")

# Test Python callable passed to C++
def calculate_premium(inputs):
    base = 500
    if inputs.get("age", 0) < 25: base += 200
    if inputs.get("property_value", 0) > 500000: base += 300
    if inputs.get("coverage_amount", 0) > 1000000: base += 500
    return base

app.set_logic(calculate_premium)

# Test Add fields maps
app.add_field("age", "Client Age", "A")
app.add_field("property_value", "Property Value ($)", "B")
app.add_field("coverage_amount", "Coverage Amount ($)", "C")
app.set_premium_column("D", "Final Premium")

print("Successfully wrapped and mapped engine fields.")

# Test Execution headless
print("Executing risk engine logic...")
test_inputs = {"age": 24, "property_value": 600000, "coverage_amount": 1500000}
result_premium = app.calculate_headless(test_inputs)
# Expected: 500 + 200 (age<25) + 300 (prop>500k) + 500 (coverage>1M) = 1500
print(f"Computed Premium: {result_premium}")
assert result_premium == 1500, "Math logic calculation failed"

# Test Excel Generation
print("Testing Excel export...")
output_file = "test_policy_quote.xlsx"
if os.path.exists(output_file):
    os.remove(output_file)

app.export_excel_headless(output_file)

# Assert file exists
assert os.path.exists(output_file), f"Excel file {output_file} was not generated!"
print(f"Success! file {output_file} generated and the wrapper is fully verified.")
