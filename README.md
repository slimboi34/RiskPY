# RiskPY: Enterprise Actuarial Engine 🚀

A hyper-fast, C++ powered pricing framework built explicitly for actuaries across all lines of insurance (P&C, Life, Health). Includes a gorgeous desktop GUI, native batch Excel exporting, and sub-millisecond Monte Carlo simulations.

> [!TIP]
> **New in v0.2.3**: The core Actuarial Math engines have been fully refactored into a stateful Object-Oriented Hierarchy! Math models are now instantiable configurations reducing duplicated arguments. They have also been secured with strict bounds checking, mathematical limit validation, and a comprehensive `pytest` regression TDD suite. The batch processor continues to naturally stream thousands of policy quotes to Excel instantly.

## The Painkiller
Currently, actuaries are forced to write complex, slow `if/else` ladders in Python or rely on clunky Excel spreadsheets to process rating models and massive stochastic simulations. `RiskPY` solves this by providing a declarative, high-performance C++ backend combined with a gorgeous built-in Desktop UI.

You write clean, readable pricing rules in Python. We execute them instantly in C++.

---

## Installation & Development

```bash
# Standard install
pip install open-riskpy

# Developer install (includes pytest harness for core modifications)
git clone https://github.com/slimboi34/RiskPY.git
cd RiskPY
pip install -e ".[dev]"
pytest tests/
```

> [!IMPORTANT]
> The Python wrapper bindings require `pybind11` and a `C++17` compiler. `scikit-build-core` handles everything automatically during installation.

---

## Core Features
1. **FactorModel** — Declarative multiplicative rating engine (P&C, Auto, Property, Liability)
2. **MonteCarloSimulator** — Stochastic simulation (Poisson frequency × Lognormal severity)
3. **ActuarialMath** — Present value, future value, loss ratios, mortality lookups
4. **LossTriangle** — Chain ladder reserving with IBNR calculation
5. **ExperienceRating** — Bühlmann credibility, NCCI experience modification
6. **ExposureRating** — Increased limits factors, excess layer pricing, burning cost
7. **RateAnalyzer** — On-level factors, rate filings, combined ratios, trend projection
8. **UnderwritingApp** — Built-in Tkinter GUI with Monte Carlo visualisation
9. **Batch Processing** — **[NEW]** Multi-record CSV-to-Excel pipeline via C++ array buffering.

---

## Comprehensive Usability Guide

### 1. Actuarial Math (Core Rules Engine)
All math modules are heavily guarded against invalid inputs (division by zero, negative intervals). Exceptions will be explicitly thrown rather than silencing math errors.

```python
from riskpy import ActuarialMath

pv = ActuarialMath.present_value(rate=0.05, periods=20, payment=50000)
fv = ActuarialMath.future_value(rate=0.07, periods=30, payment=10000)
lr = ActuarialMath.calculate_loss_ratio(incurred_losses=750000, earned_premium=1000000)
```

### 2. FactorModel (Declarative Pricing)
Replace your nested Python logic blocks with our Sub-Millisecond Multiplier Engine. 

```python
from riskpy import FactorModel

model = FactorModel(initial_base_rate=1000.0)

# Add string-exact matching rules
model.add_multiplier("state", "FL", 3.0)
model.add_multiplier("state", "CA", 2.0)

# Add boundary thresholds for continuous variables
model.add_numeric_band_multiplier("age", 16, 25, 2.0)
model.add_numeric_band_multiplier("age", 26, 65, 1.0)

premium = model.calculate({"state": "FL", "age": 19.0})
# Computed locally in C++: $1,000 × 3.0 × 2.0 = $6,000.00
```

### 3. Native Batch Processing & Excel Integration
> [!NOTE]  
> The new `UnderwritingApp::calculate_batch` aggregates inputs and premium decisions directly into memory, before making a single, massive C++ instruction call to build your `batch_policy_quotes.xlsx` natively.

```python
from riskpy import UnderwritingApp, FactorModel

app = UnderwritingApp(title="Auto Pricing Actuary", excel_template="template.xlsx")

# 1. Define fields & what column they map to in Excel
app.add_field("state", "Driver State", "A", choices=["NY", "CA", "FL"])
app.add_field("age", "Driver Age", "B")
app.set_premium_column("C", "Calculated Premium")

# 2. Assign the C++ rules logic
app.set_factor_model(FactorModel(100.0))

# 3. Stream a 50,000-line CSV and instantly dump to Excel
total_book_premium, success_count = app.calculate_batch("historical_book.csv", "batch_output.xlsx")
print(f"Priced {success_count} quotes for ${total_book_premium:,.2f} Total Premium")
```

### 4. Monte Carlo Simulator
Blazing fast distributions processed over millions of trials. Safe bounds checking ensures standard deviations and mu limits are never negative. Now supporting Geometric Brownian Motion, Gamma-Health Claims, and Pareto Catastrophe Modeling!

```python
from riskpy import MonteCarloSimulator
import numpy as np

sim = MonteCarloSimulator(trials=100000)

# Example 1: Property & Casualty Aggregate Loss (Poisson Frequency x Lognormal Severity)
losses = sim.simulate_aggregate_loss(
    expected_frequency=5.0, expected_severity_mu=10.0, severity_sigma=1.5
)
print(f"99% P&C VaR: ${np.percentile(losses, 99):,.0f}")

# Example 2: Economic Scenarios (Geometric Brownian Motion) 
asset_values = sim.simulate_economic_path(
    initial_price=100.0, drift=0.08, volatility=0.15, time_steps=10
)
print(f"Mean Asset Value over 10 Years: ${np.mean(asset_values):,.2f}")
```

### 5. Exposure & Experience Rating
Complete P&C specific mathematical paradigms.

```python
from riskpy import ExperienceRating, ExposureRating

# NCCI Workers' Comp Mod Calculation
# Configure the model parameters once
er = ExperienceRating(k_parameter=1082.0, ballast=0.3)
emf = er.experience_mod_factor(actual_losses=800000, expected_losses=1000000)

# ILF Pareto Curve (for Excess Layers)
# Establish the layer characteristics
layer = ExposureRating(base_limit=100000, b_parameter=0.6)
ilf = layer.increased_limits_factor(target_limit=1000000)
```

---

## TDD & Security Assurances
In version `0.2.3`, we migrated to a strict **Test-Driven Development (TDD)** environment utilizing `pytest` including a full OOP baseline regression sequence.

- **Division By Zero**: Eradicated. Functions like `RateAnalyzer.required_rate_change` mathematically secure denominator safety against 0 and NaN.
- **Negative Bounds Validation**: Functions natively intercept impossible physical geometries (e.g. passing a negative claim count to `ExperienceRating.calculate_credibility`).
- **Array Mismatching**: Multi-year arrays (like `rate_changes` vs `earned_premiums`) are inherently aligned before looping to prevent C++ core dumps and `SEGFAULT` issues.

To execute the verification suite:
```bash
pytest tests/
```

## License
MIT — Free for personal and commercial use.

- **GitHub**: [github.com/slimboi34/RiskPY](https://github.com/slimboi34/RiskPY)
- **PyPI**: [pypi.org/project/open-riskpy](https://pypi.org/project/open-riskpy)
