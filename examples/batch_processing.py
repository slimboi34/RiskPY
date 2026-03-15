"""
RiskPY Example: Batch CSV Processing
======================================
This example shows how to process thousands of policy 
submissions from a CSV file and export results to Excel.

Run this file:
    python batch_processing.py
"""
import csv
import os
import random
from riskpy import UnderwritingApp, FactorModel

# =============================================
# 1. CREATE A SAMPLE CSV (simulating broker submission)
# =============================================
csv_file = "sample_submissions.csv"
states = ["NY", "CA", "FL", "TX", "OH", "IL", "PA", "GA"]
vehicles = ["Sedan", "SUV", "Sports Car", "Pickup Truck"]
claims = ["Clean", "1 Claim", "2+ Claims"]

print("Generating 500 random policy submissions...")
with open(csv_file, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["state", "driver_age", "vehicle_type", "claims_history"])
    writer.writeheader()
    for _ in range(500):
        writer.writerow({
            "state": random.choice(states),
            "driver_age": random.randint(16, 80),
            "vehicle_type": random.choice(vehicles),
            "claims_history": random.choice(claims),
        })
print(f"  Created {csv_file} with 500 records")

# =============================================
# 2. CONFIGURE THE RATING MODEL
# =============================================
model = FactorModel(initial_base_rate=750.0)

# State factors
for state, factor in [("NY",1.8),("CA",2.0),("FL",2.5),("TX",1.4),("OH",1.0),("IL",1.3),("PA",1.2),("GA",1.1)]:
    model.add_multiplier("state", state, factor)

# Age bands
model.add_numeric_band_multiplier("driver_age", 16, 25, 2.0)
model.add_numeric_band_multiplier("driver_age", 26, 55, 1.0)
model.add_numeric_band_multiplier("driver_age", 56, 99, 1.3)

# Vehicle factors
for vtype, factor in [("Sedan",1.0),("SUV",1.2),("Sports Car",2.0),("Pickup Truck",1.1)]:
    model.add_multiplier("vehicle_type", vtype, factor)

# Claims factors
model.add_multiplier("claims_history", "Clean", 0.8)
model.add_multiplier("claims_history", "1 Claim", 1.2)
model.add_multiplier("claims_history", "2+ Claims", 1.8)

# =============================================
# 3. BUILD THE APP (headless mode)
# =============================================
app = UnderwritingApp(title="Batch Processor")
app.add_field("state", "State", "A", choices=states)
app.add_field("driver_age", "Driver Age", "B")
app.add_field("vehicle_type", "Vehicle Type", "C", choices=vehicles)
app.add_field("claims_history", "Claims History", "D", choices=claims)
app.set_factor_model(model)

# =============================================
# 4. PROCESS THE ENTIRE CSV
# =============================================
print("\nProcessing all 500 submissions through C++ engine...")
total_premium, count = app.calculate_batch(csv_file, "rated_submissions.xlsx")

print(f"\n--- Batch Results ---")
print(f"  Records processed:   {count:>8,}")
print(f"  Total book premium:  ${total_premium:>12,.2f}")
print(f"  Average premium:     ${total_premium/count:>12,.2f}")
print(f"  Output file:         rated_submissions.xlsx")

# Cleanup
os.remove(csv_file)
print(f"\nDone! Open rated_submissions.xlsx to see all {count} rated policies.")
