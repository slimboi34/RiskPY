"""
RiskPY Example: Catastrophe Risk Modeling
==========================================
This example demonstrates how to use MonteCarloSimulator
for natural catastrophe modeling and reinsurance pricing.

Simulates hurricane losses for a P&C portfolio using:
- Poisson distribution for claim frequency
- Lognormal distribution for claim severity

Run this file:
    python catastrophe_model.py
"""
from riskpy import MonteCarloSimulator
import numpy as np

print("=" * 60)
print("  RiskPY — Catastrophe Risk Model")
print("=" * 60)

# =============================================
# SCENARIO 1: MODERATE HURRICANE SEASON
# =============================================
# Expected 3 hurricane landfalls per year
# Average loss per event: ~$50M (mu=17.7, sigma=0.8 in log-space)

print("\n--- Scenario 1: Moderate Hurricane Season ---")

moderate = MonteCarloSimulator.simulate_aggregate_loss(
    trials=200000,
    expected_frequency=3.0,
    expected_severity_mu=17.7,    # ln($50M) ≈ 17.7
    severity_sigma=0.8
)

print(f"  Expected annual loss:  ${np.mean(moderate):>15,.0f}")
print(f"  Median annual loss:    ${np.median(moderate):>15,.0f}")
print(f"  90% VaR:               ${np.percentile(moderate, 90):>15,.0f}")
print(f"  95% VaR:               ${np.percentile(moderate, 95):>15,.0f}")
print(f"  99% VaR:               ${np.percentile(moderate, 99):>15,.0f}")
print(f"  99.5% VaR (1-in-200):  ${np.percentile(moderate, 99.5):>15,.0f}")

# =============================================
# SCENARIO 2: SEVERE HURRICANE SEASON
# =============================================
# Expected 7 events, higher severity volatility

print("\n--- Scenario 2: Severe Hurricane Season ---")

severe = MonteCarloSimulator.simulate_aggregate_loss(
    trials=200000,
    expected_frequency=7.0,
    expected_severity_mu=18.4,    # ln($100M) ≈ 18.4
    severity_sigma=1.2
)

print(f"  Expected annual loss:  ${np.mean(severe):>15,.0f}")
print(f"  Median annual loss:    ${np.median(severe):>15,.0f}")
print(f"  90% VaR:               ${np.percentile(severe, 90):>15,.0f}")
print(f"  95% VaR:               ${np.percentile(severe, 95):>15,.0f}")
print(f"  99% VaR:               ${np.percentile(severe, 99):>15,.0f}")
print(f"  99.5% VaR (1-in-200):  ${np.percentile(severe, 99.5):>15,.0f}")

# =============================================
# REINSURANCE ANALYSIS
# =============================================
# If the insurer buys reinsurance with a $500M attachment point,
# how often does the reinsurance layer get triggered?

attachment = 500_000_000
breaches_moderate = np.sum(np.array(moderate) > attachment) / len(moderate) * 100
breaches_severe = np.sum(np.array(severe) > attachment) / len(severe) * 100

print(f"\n--- Reinsurance Layer Analysis (${attachment/1e6:.0f}M attachment) ---")
print(f"  Moderate scenario breach frequency: {breaches_moderate:.2f}%")
print(f"  Severe scenario breach frequency:   {breaches_severe:.2f}%")

# =============================================
# COMPARISON TABLE
# =============================================
print(f"\n--- Scenario Comparison ---")
print(f"  {'Metric':<25} {'Moderate':>15} {'Severe':>15}")
print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*15}")
print(f"  {'Expected Loss':<25} ${np.mean(moderate):>14,.0f} ${np.mean(severe):>14,.0f}")
print(f"  {'99% VaR':<25} ${np.percentile(moderate,99):>14,.0f} ${np.percentile(severe,99):>14,.0f}")
print(f"  {'Max Simulated':<25} ${np.max(moderate):>14,.0f} ${np.max(severe):>14,.0f}")

print(f"\n{'=' * 60}")
print(f"  200,000 trials per scenario executed in C++ (<1 second)")
print(f"{'=' * 60}")
