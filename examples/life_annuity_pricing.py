"""
RiskPY Example: Life Annuity Pricing
=====================================
This example demonstrates how to use ActuarialMath and 
MonteCarloSimulator for life insurance and annuity calculations.

Covers:
- Present Value of annuity payments
- Future Value of premium accumulation
- Mortality rate lookups
- Stochastic mortality simulation for a portfolio

Run this file:
    python life_annuity_pricing.py
"""
from riskpy import ActuarialMath, MonteCarloSimulator
import numpy as np

print("=" * 60)
print("  RiskPY — Life & Annuity Pricing Demo")
print("=" * 60)

# =============================================
# 1. PRESENT VALUE OF AN ANNUITY
# =============================================
# A retiree wants to receive $60,000 per year for 25 years.
# The insurer uses a 4.5% discount rate.
# Question: What lump sum must the insurer hold today?

annual_payment = 60000
discount_rate = 0.045
payout_years = 25

pv = ActuarialMath.present_value(
    rate=discount_rate, 
    periods=payout_years, 
    payment=annual_payment
)

print(f"\n--- Annuity Present Value ---")
print(f"  Annual payment:  ${annual_payment:>12,.2f}")
print(f"  Discount rate:   {discount_rate:>12.1%}")
print(f"  Payout period:   {payout_years:>12} years")
print(f"  Present Value:   ${pv:>12,.2f}")
print(f"  → The insurer needs ${pv:,.0f} today to fund this annuity")

# =============================================
# 2. FUTURE VALUE OF PREMIUM ACCUMULATION
# =============================================
# A policyholder pays $5,000/year in premiums for 30 years.
# The insurer invests at 6% annual return.
# Question: How much will the accumulated fund be worth?

premium = 5000
investment_rate = 0.06
accumulation_years = 30

fv = ActuarialMath.future_value(
    rate=investment_rate,
    periods=accumulation_years,
    payment=premium
)

print(f"\n--- Premium Accumulation ---")
print(f"  Annual premium:  ${premium:>12,.2f}")
print(f"  Investment rate:  {investment_rate:>12.1%}")
print(f"  Accumulation:    {accumulation_years:>12} years")
print(f"  Future Value:    ${fv:>12,.2f}")
print(f"  → Total premiums paid: ${premium * accumulation_years:,.0f}")
print(f"  → Investment gain:     ${fv - (premium * accumulation_years):,.0f}")

# =============================================
# 3. MORTALITY RATE LOOKUPS
# =============================================
# Look up qx (probability of death within 1 year) for key ages.

print(f"\n--- CSO Mortality Table (qx) ---")
print(f"  {'Age':>5} | {'qx':>10} | {'Interpretation'}")
print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*30}")

for age in [25, 35, 45, 55, 65, 75, 85]:
    qx = ActuarialMath.lookup_mortality_rate(age)
    per_thousand = qx * 1000
    print(f"  {age:>5} | {qx:>10.6f} | {per_thousand:.2f} deaths per 1,000 lives")

# =============================================
# 4. LOSS RATIO ANALYSIS
# =============================================
incurred = 8500000
earned = 12000000
lr = ActuarialMath.calculate_loss_ratio(incurred, earned)

print(f"\n--- Loss Ratio ---")
print(f"  Incurred losses:  ${incurred:>12,.0f}")
print(f"  Earned premium:   ${earned:>12,.0f}")
print(f"  Loss ratio:       {lr:>12.1%}")
if lr < 0.6:
    print(f"  → Highly profitable book")
elif lr < 0.8:
    print(f"  → Acceptable profitability")
else:
    print(f"  → Under-priced — consider rate increase")

# =============================================
# 5. STOCHASTIC MORTALITY SIMULATION
# =============================================
# Simulate a portfolio of 10,000 life policies.
# Base mortality rate: 0.2% (qx = 0.002)
# Shock volatility: 30% (models pandemic/catastrophe risk)
# Death benefit: $250,000 per policy

print(f"\n--- Monte Carlo Life Portfolio Simulation ---")

results = MonteCarloSimulator.simulate_life_portfolio(
    trials=50000,
    policy_count=10000,
    base_mortality_rate=0.002,
    shock_volatility=0.3,
    death_benefit=250000.0
)

expected = np.mean(results)
std_dev = np.std(results)
var_95 = np.percentile(results, 95)
var_99 = np.percentile(results, 99)
worst = np.max(results)

print(f"  Portfolio size:   {10000:>12,} policies")
print(f"  Death benefit:    ${250000:>12,.0f}")
print(f"  Simulated trials: {50000:>12,}")
print(f"  Expected claims:  ${expected:>12,.0f}")
print(f"  Std deviation:    ${std_dev:>12,.0f}")
print(f"  95% VaR:          ${var_95:>12,.0f}")
print(f"  99% VaR:          ${var_99:>12,.0f}")
print(f"  Worst scenario:   ${worst:>12,.0f}")
print(f"  → Reserve at 99% VaR = ${var_99:,.0f}")

print(f"\n{'=' * 60}")
print(f"  Demo complete. All calculations powered by C++ backend.")
print(f"{'=' * 60}")
