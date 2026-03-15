# Actuarial Use Cases

This library was built to tackle the three main pillars of actuarial pricing and reserving:

## 1. Declarative Pricing (Property & Casualty / General)
**Use Case**: You are building a pricing rater for Auto Insurance. You need a base rate, and you need to multiply that rate based on categorical factors (State) and continuous bands (Age).
**The Pain**: Writing `if state == "CA": premium *= 2.0` across 50 states and thousands of rows is slow and error-prone.
**The Solution: `FactorModel`**
```python
from riskpy import FactorModel

model = FactorModel(initial_base_rate=1000.0)
# Categorical Strings
model.add_multiplier("vehicle_type", "Sedan", 1.0)
model.add_multiplier("vehicle_type", "Sports Car", 2.5)

# Continuous Numeric Bands
model.add_numeric_band_multiplier("driver_age", 16, 25, 2.0)
model.add_numeric_band_multiplier("driver_age", 26, 99, 1.0)
```

## 2. Stochastic Tail Risk (Reinsurance / Capital Modeling)
**Use Case**: You need to calculate the 99% Value-at-Risk (VaR) for a catastrophe book. You expect 5 hurricanes a year (Poisson) with an average cost of $10M each (Lognormal).
**The Pain**: Python's `random` or even `NumPy` can struggle when managing massive nested severity/frequency loops efficiently inside a GUI.
**The Solution: `MonteCarloSimulator`**
```python
from riskpy import MonteCarloSimulator

# Runs 1,000,000 simulations instantly in C++ using true Hardware Randomness
# Returns a Python array of the aggregated losses for plotting
results = MonteCarloSimulator.simulate_aggregate_loss(
    trials=1000000, 
    expected_frequency=5.0, 
    expected_severity_mu=10.0, 
    severity_sigma=1.5
)
```

## 3. Core Math (Life & Annuity)
**Use Case**: You need to rapidly discount cash flows or look up mortality decrements to price a life annuity.
**The Pain**: Re-writing robust financial formulas across different scripts.
**The Solution: `ActuarialMath`**
```python
from riskpy import ActuarialMath

# $50k yearly payment for 20 years at 5% discount rate
pv = ActuarialMath.present_value(rate=0.05, periods=20, payment=50000)

# Pull CSO mortality rate for an 85 year old
q_x = ActuarialMath.lookup_mortality_rate(age=85)
```
