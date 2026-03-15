# ==========================================
# 1. IMPORT WRAPPER
# ==========================================
from riskpy import UnderwritingApp, FactorModel, ActuarialMath

# ==========================================
# 2. CONFIGURE TOOL
# ==========================================
app = UnderwritingApp(title="Actuarial Pricing Tool", excel_template="corporate_layout.xlsx")

# Add Fields automatically maps: C++ Field Engine -> Tkinter UI -> Excel Output
app.add_field(name="state", label="Location State", excel_col="A", choices=["NY", "CA", "TX", "FL"])
app.add_field(name="age", label="Client Age", excel_col="B")
app.add_field(name="property_value", label="Property Value ($)", excel_col="C")
app.add_field(name="coverage_amount", label="Coverage Amount ($)", excel_col="D")
app.set_premium_column(excel_col="E", label="Final Premium")

# ==========================================
# 3. ACTUARIAL FACTOR MODEL (100% C++ Engine Execution)
# ==========================================
# Actuaries don't need to write Python `if` statements anymore.
# They just build rating tables declaratively:
model = FactorModel(initial_base_rate=500.0)

# State Class Factors
model.add_multiplier("state", "CA", 2.0)   # California has high risk, double the base 
model.add_multiplier("state", "FL", 3.0)   # Florida hurricane risk
model.add_multiplier("state", "NY", 1.4)

# Age Class Band Factors
model.add_numeric_band_multiplier("age", 18, 25, 1.4) # Young clients are a higher risk

# Coverage Tier Factors
model.add_numeric_band_multiplier("coverage_amount", 1000000, 5000000, 2.0) # Million dollar payouts double the premium
model.add_numeric_band_multiplier("property_value", 500000, 10000000, 1.6)

app.set_factor_model(model)

# Bonus: Proving out the built-in standard Actuarial computations
pv = ActuarialMath.present_value(0.05, 10, 1000)
print(f"System Check: Built in C++ computation for Present Value: ${pv:,.2f}")

# ==========================================
# 4. RUN THE GUI
# ==========================================
if __name__ == "__main__":
    app.run(export_filename="policy_quote.xlsx")
