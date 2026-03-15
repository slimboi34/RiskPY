"""
RiskPY Example: Auto Insurance Rating Engine
=============================================
This example demonstrates how to build a complete auto insurance
pricing model using the FactorModel declarative engine.

The model applies multiplicative factors for:
- US State (territorial rating)
- Driver Age (age banding)
- Vehicle Type (class rating)
- Claims History (experience modification)

Run this file:
    python auto_insurance_rater.py
"""
from riskpy import UnderwritingApp, FactorModel

# =============================================
# 1. DEFINE THE BASE RATE
# =============================================
# The base rate is the starting premium before any factors.
# In practice, this comes from actuarial loss cost analysis.
model = FactorModel(initial_base_rate=800.0)

# =============================================
# 2. TERRITORIAL FACTORS (by US State)
# =============================================
# Each state has different loss experience due to weather, 
# litigation environment, and traffic density.
model.add_multiplier("state", "NY", 1.8)   # High traffic density
model.add_multiplier("state", "CA", 2.0)   # High cost of living + wildfire risk
model.add_multiplier("state", "FL", 2.5)   # Hurricane + uninsured motorists
model.add_multiplier("state", "TX", 1.4)   # Moderate risk
model.add_multiplier("state", "OH", 1.0)   # Base territory

# =============================================
# 3. AGE BAND FACTORS
# =============================================
# Younger and older drivers have higher claim frequencies.
# These bands reflect actuarial loss curves by age.
model.add_numeric_band_multiplier("driver_age", 16, 20, 2.5)   # Teen drivers
model.add_numeric_band_multiplier("driver_age", 21, 25, 1.8)   # Young adults
model.add_numeric_band_multiplier("driver_age", 26, 35, 1.0)   # Prime risk
model.add_numeric_band_multiplier("driver_age", 36, 55, 0.9)   # Experienced
model.add_numeric_band_multiplier("driver_age", 56, 65, 1.0)   # Standard
model.add_numeric_band_multiplier("driver_age", 66, 99, 1.4)   # Senior

# =============================================
# 4. VEHICLE CLASS FACTORS
# =============================================
model.add_multiplier("vehicle_type", "Sedan", 1.0)
model.add_multiplier("vehicle_type", "SUV", 1.2)
model.add_multiplier("vehicle_type", "Sports Car", 2.0)
model.add_multiplier("vehicle_type", "Pickup Truck", 1.1)
model.add_multiplier("vehicle_type", "Electric Vehicle", 0.9)

# =============================================
# 5. CLAIMS HISTORY FACTORS
# =============================================
model.add_multiplier("claims_history", "Clean", 0.8)       # Discount for no claims
model.add_multiplier("claims_history", "1 Claim", 1.2)
model.add_multiplier("claims_history", "2+ Claims", 1.8)

# =============================================
# 6. BUILD THE GUI
# =============================================
app = UnderwritingApp(title="Auto Insurance Rater — RiskPY Demo")

app.add_field("state", "State", "A", 
              choices=["NY", "CA", "FL", "TX", "OH"])
app.add_field("driver_age", "Driver Age", "B")
app.add_field("vehicle_type", "Vehicle Type", "C", 
              choices=["Sedan", "SUV", "Sports Car", "Pickup Truck", "Electric Vehicle"])
app.add_field("claims_history", "Claims History", "D", 
              choices=["Clean", "1 Claim", "2+ Claims"])

app.set_factor_model(model)

# =============================================
# 7. LAUNCH
# =============================================
# This opens a dual-tab GUI:
#   Tab 1: Interactive pricing form
#   Tab 2: Monte Carlo simulation dashboard
print("Launching Auto Insurance Rater...")
print("Example: A 19-year-old in Florida driving a Sports Car with 2+ claims:")
example_premium = model.calculate({
    "state": "FL", 
    "driver_age": 19.0, 
    "vehicle_type": "Sports Car", 
    "claims_history": "2+ Claims"
})
print(f"  Premium: ${example_premium:,.2f}")
print(f"  Breakdown: $800 base × 2.5 (FL) × 2.5 (teen) × 2.0 (sports) × 1.8 (claims)")
print()

app.run()
