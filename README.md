# RiskPY

A hyper-fast, C++ powered pricing framework built explicitly for actuaries across all lines of insurance (P&C, Life, Health).

## The Painkiller
Currently, actuaries are forced to write complex, slow `if/else` ladders in Python or rely on clunky Excel spreadsheets to process rating models and massive stochastic simulations. `RiskPY` solves this by providing a declarative, high-performance C++ backend combined with a gorgeous built-in Desktop UI.

You write clean, readable pricing rules in Python. We execute them instantly in C++.

## Installation
```bash
pip install open-riskpy
```

## Features
- **FactorModel** — Declarative multiplicative rating engine (P&C, Auto, Property, Liability)
- **MonteCarloSimulator** — Stochastic simulation (Poisson frequency × Lognormal severity)
- **ActuarialMath** — Present value, future value, loss ratios, mortality lookups
- **LossTriangle** — Chain ladder reserving with IBNR calculation
- **ExperienceRating** — Bühlmann credibility, NCCI experience modification
- **ExposureRating** — Increased limits factors, excess layer pricing, burning cost
- **RateAnalyzer** — On-level factors, rate filings, combined ratios, trend projection
- **UnderwritingApp** — Built-in Tkinter GUI with Monte Carlo visualisation
- **Batch Processing** — CSV-to-Excel pipeline via C++ (bypasses Python GIL)

---

# Quick Start

### 1. Declarative Pricing
```python
from riskpy import FactorModel

model = FactorModel(initial_base_rate=1000.0)
model.add_multiplier("state", "FL", 3.0)
model.add_numeric_band_multiplier("age", 16, 25, 2.0)

premium = model.calculate({"state": "FL", "age": 19.0})
# Result: $1,000 × 3.0 × 2.0 = $6,000.00
```

### 2. Monte Carlo Simulation
```python
from riskpy import MonteCarloSimulator
import numpy as np

losses = MonteCarloSimulator.simulate_aggregate_loss(
    trials=100000, expected_frequency=5.0,
    expected_severity_mu=10.0, severity_sigma=1.5
)
print(f"99% VaR: ${np.percentile(losses, 99):,.0f}")
```

### 3. Chain Ladder Reserving
```python
from riskpy import LossTriangle

triangle = LossTriangle()
triangle.add_origin_year(2020, [1000, 1500, 1800, 1900])
triangle.add_origin_year(2021, [1100, 1600, 1850])
triangle.add_origin_year(2022, [1200, 1700])
triangle.add_origin_year(2023, [1300])

factors = triangle.get_development_factors()
ultimates = triangle.get_ultimate_losses()
ibnr = triangle.get_ibnr_reserves()
print(f"IBNR Reserves: {[f'${x:,.0f}' for x in ibnr]}")
```

### 4. Experience Rating & Credibility
```python
from riskpy import ExperienceRating

z = ExperienceRating.calculate_credibility(claim_count=500, k_parameter=1082)
blended = ExperienceRating.credibility_weighted_rate(
    observed_rate=0.08, expected_rate=0.05, credibility_z=z
)
emf = ExperienceRating.experience_mod_factor(
    actual_losses=800000, expected_losses=1000000, ballast=0.3
)
print(f"Credibility: {z:.2f}, Blended Rate: {blended:.4f}, EMF: {emf:.3f}")
```

### 5. Excess Layer Pricing
```python
from riskpy import ExposureRating

ilf = ExposureRating.increased_limits_factor(
    base_limit=100000, target_limit=1000000, b_parameter=0.6
)
cost = ExposureRating.burning_cost(
    historical_losses=[50000, 250000, 800000, 1200000, 30000],
    attachment=200000, limit=500000, exposure_years=5
)
print(f"ILF: {ilf:.3f}, Burning Cost: ${cost:,.0f}")
```

### 6. Rate Filing Analysis
```python
from riskpy import RateAnalyzer

olf = RateAnalyzer.on_level_factor([0.05, 0.03, 0.10])
cr = RateAnalyzer.combined_ratio(loss_ratio=0.72, expense_ratio=0.28)
needed = RateAnalyzer.required_rate_change(
    target_combined_ratio=0.95, current_loss_ratio=0.72, expense_ratio=0.28
)
trend = RateAnalyzer.trend_factor(annual_trend=0.04, years=3.0)
print(f"On-Level Factor: {olf:.4f}, CR: {cr:.1%}, Needed Change: {needed:.1%}")
```

### 7. Full GUI Application
```python
from riskpy import UnderwritingApp, FactorModel

app = UnderwritingApp(title="Actuarial Pricing Tool")
app.add_field("state", "State", "A", choices=["NY", "CA", "FL"])
app.add_field("age", "Client Age", "B")

model = FactorModel(initial_base_rate=500.0)
model.add_multiplier("state", "CA", 2.0)
app.set_factor_model(model)
app.run()  # Launches desktop GUI with Pricing + Monte Carlo tabs
```

### 8. Headless Batch Processing
```python
total_premium, count = app.calculate_batch("policies.csv", "output.xlsx")
print(f"Processed {count} policies. Total: ${total_premium:,.2f}")
```

---

# Full API Reference

## FactorModel

A declarative, multiplicative rating engine. Define pricing rules as factors — the C++ engine applies them instantly.

```python
from riskpy import FactorModel
model = FactorModel(initial_base_rate=1000.0)
```

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `set_base_rate(rate)` | `float` | `None` | Updates the base rate |
| `add_multiplier(field_name, exact_value, multiplier)` | `str, str, float` | `None` | Adds a categorical factor rule |
| `add_numeric_band_multiplier(field_name, min_val, max_val, multiplier)` | `str, float, float, float` | `None` | Adds a continuous band factor |
| `calculate(inputs)` | `dict[str, str\|float]` | `float` | Calculates premium with all matching rules applied |

**Example — Auto Insurance:**
```python
model = FactorModel(initial_base_rate=800.0)
model.add_multiplier("state", "CA", 2.0)         # California territory
model.add_multiplier("state", "FL", 2.5)         # Florida territory
model.add_multiplier("vehicle", "Sports Car", 2.0)
model.add_numeric_band_multiplier("age", 16, 25, 2.0)  # Young driver
model.add_numeric_band_multiplier("age", 26, 65, 1.0)  # Standard
model.add_numeric_band_multiplier("age", 66, 99, 1.3)  # Senior

premium = model.calculate({"state": "FL", "age": 19.0, "vehicle": "Sports Car"})
# $800 × 2.5 × 2.0 × 2.0 = $8,000.00
```

---

## MonteCarloSimulator

High-performance stochastic simulation running natively in C++. All methods are static.

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `simulate_aggregate_loss(trials, expected_frequency, expected_severity_mu, severity_sigma)` | `int, float, float, float` | `list[float]` | P&C aggregate loss simulation (Poisson × Lognormal) |
| `simulate_life_portfolio(trials, policy_count, base_mortality_rate, shock_volatility, death_benefit)` | `int, int, float, float, float` | `list[float]` | Life portfolio mortality simulation with systemic shocks |

**Example — Catastrophe Model:**
```python
from riskpy import MonteCarloSimulator
import numpy as np

results = MonteCarloSimulator.simulate_aggregate_loss(
    trials=200000, expected_frequency=5.0,
    expected_severity_mu=17.7, severity_sigma=1.2
)
print(f"Expected: ${np.mean(results):,.0f}")
print(f"99.5% VaR (1-in-200): ${np.percentile(results, 99.5):,.0f}")
```

**Example — Life Portfolio Stress Test:**
```python
results = MonteCarloSimulator.simulate_life_portfolio(
    trials=50000, policy_count=10000,
    base_mortality_rate=0.002, shock_volatility=0.3,
    death_benefit=500000.0
)
print(f"99% VaR: ${np.percentile(results, 99):,.0f}")
```

---

## ActuarialMath

Core actuarial formulas in C++. All methods are static.

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `present_value(rate, periods, payment)` | `float, int, float` | `float` | Present value of an ordinary annuity |
| `future_value(rate, periods, payment)` | `float, int, float` | `float` | Future value of an ordinary annuity |
| `calculate_loss_ratio(incurred_losses, earned_premium)` | `float, float` | `float` | Loss ratio (key P&C KPI) |
| `lookup_mortality_rate(age)` | `int` | `float` | CSO mortality table lookup (qx) |

```python
from riskpy import ActuarialMath

pv = ActuarialMath.present_value(rate=0.05, periods=20, payment=50000)
fv = ActuarialMath.future_value(rate=0.07, periods=30, payment=10000)
lr = ActuarialMath.calculate_loss_ratio(750000, 1000000)  # 75%
qx = ActuarialMath.lookup_mortality_rate(age=65)
```

---

## LossTriangle

Chain ladder loss development for reserving. Used by every P&C reserving department.

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `add_origin_year(year, cumulative_payments)` | `int, list[float]` | `None` | Add a row of cumulative payments |
| `get_development_factors()` | — | `list[float]` | Weighted age-to-age link ratios |
| `get_ultimate_losses()` | — | `list[float]` | Projected ultimate losses |
| `get_ibnr_reserves()` | — | `list[float]` | IBNR reserves (ultimate − latest) |
| `get_origin_year_count()` | — | `int` | Number of origin years |
| `get_development_period_count()` | — | `int` | Number of development periods |

```python
from riskpy import LossTriangle

tri = LossTriangle()
tri.add_origin_year(2019, [2000, 3500, 4200, 4500, 4550])
tri.add_origin_year(2020, [2200, 3800, 4600, 4900])
tri.add_origin_year(2021, [2500, 4100, 5000])
tri.add_origin_year(2022, [2700, 4400])
tri.add_origin_year(2023, [3000])

print(f"Development Factors: {tri.get_development_factors()}")
print(f"Ultimate Losses:     {tri.get_ultimate_losses()}")
print(f"IBNR Reserves:       {tri.get_ibnr_reserves()}")
```

---

## ExperienceRating

Credibility theory and experience modification for large account pricing.

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `credibility_weighted_rate(observed, expected, z)` | `float, float, float` | `float` | Bühlmann credibility blending |
| `calculate_credibility(claim_count, k_parameter)` | `int, float` | `float` | Classical limited fluctuation Z |
| `experience_mod_factor(actual, expected, ballast)` | `float, float, float` | `float` | NCCI-style experience modification |
| `buhlmann_straub_credibility(exposure, variance_ratio)` | `float, float` | `float` | Bühlmann-Straub Z factor |
| `expected_losses(exposure, frequency, severity)` | `float, float, float` | `float` | Expected loss calculation |

```python
from riskpy import ExperienceRating

# Classical credibility (full credibility at 1082 claims)
z = ExperienceRating.calculate_credibility(claim_count=500, k_parameter=1082)

# Blend individual experience with class average
rate = ExperienceRating.credibility_weighted_rate(0.08, 0.05, z)

# Workers' Comp experience mod
emf = ExperienceRating.experience_mod_factor(
    actual_losses=800000, expected_losses=1000000, ballast=0.3
)
```

---

## ExposureRating

Excess layer pricing, increased limits factors, and reinsurance analysis.

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `increased_limits_factor(base, target, b)` | `float, float, float` | `float` | Pareto ILF curve |
| `layer_premium(gup, att, lim, ilf_base, ilf_att, ilf_lim)` | `6× float` | `float` | Price an excess layer |
| `burning_cost(losses, attachment, limit, years)` | `list, float, float, int` | `float` | Historical burning cost |
| `loss_elimination_ratio(deductible, expected, cv)` | `float, float, float` | `float` | Deductible credit analysis |
| `excess_risk_load(std_dev, multiple)` | `float, float` | `float` | Risk load for excess layers |

```python
from riskpy import ExposureRating

# ILF from $100K to $1M using Pareto
ilf = ExposureRating.increased_limits_factor(100000, 1000000, 0.6)

# Burning cost for $500K xs $200K layer
cost = ExposureRating.burning_cost(
    [50000, 250000, 800000, 1200000], 200000, 500000, 4
)

# Deductible credit
ler = ExposureRating.loss_elimination_ratio(25000, 100000, 1.5)
```

---

## RateAnalyzer

Rate filing tools, on-leveling, trending, and portfolio KPIs.

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `on_level_factor(rate_changes)` | `list[float]` | `float` | Cumulative on-level factor |
| `rate_change_impact(old, new)` | `float, float` | `float` | Percentage rate change |
| `combined_ratio(lr, er)` | `float, float` | `float` | Combined ratio KPI |
| `required_rate_change(target_cr, lr, er)` | `float, float, float` | `float` | Rate change needed for target profitability |
| `on_level_premiums(premiums, changes)` | `list, list` | `list[float]` | Bring historical premiums to current level |
| `permissible_loss_ratio(target_cr, er)` | `float, float` | `float` | Target loss ratio |
| `trend_factor(trend, years)` | `float, float` | `float` | Loss trend projection |
| `loss_cost_rate(losses, exposure)` | `float, float` | `float` | Pure premium per exposure |

```python
from riskpy import RateAnalyzer

# On-level 3 years of rate changes
olf = RateAnalyzer.on_level_factor([0.05, 0.03, 0.10])

# How much rate increase do we need?
needed = RateAnalyzer.required_rate_change(
    target_combined_ratio=0.95,
    current_loss_ratio=0.72,
    expense_ratio=0.28
)

# Trend losses forward
trend = RateAnalyzer.trend_factor(annual_trend=0.04, years=3.0)

# On-level a book of earned premiums
leveled = RateAnalyzer.on_level_premiums(
    [1000000, 1100000, 1200000],
    [0.05, 0.03, 0.08]
)
```

---

## UnderwritingApp

Interactive Tkinter GUI with two tabs: Pricing & Rating, and Monte Carlo Predictor.

```python
from riskpy import UnderwritingApp, FactorModel
app = UnderwritingApp(title="My Tool", excel_template="template.xlsx")
```

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `add_field(name, label, col, choices)` | `str, str, str, list\|None` | `None` | Add GUI input field + Excel mapping |
| `set_factor_model(model)` | `FactorModel` | `None` | Connect C++ rating engine |
| `set_logic(func)` | `callable` | `None` | Set custom Python pricing function |
| `calculate_headless(inputs)` | `dict` | `float` | Run calculation without GUI |
| `calculate_batch(csv, output)` | `str, str` | `(float, int)` | Batch CSV processing |
| `run()` | — | `None` | Launch the desktop GUI |

---

## Architecture

RiskPY is a hybrid C++/Python framework using `pybind11` to bridge the two languages:

```
Python API (riskpy/)  →  pybind11 bridge  →  C++ Engine (src/)
     │                                           │
  Tkinter GUI                              FactorModel.cpp
  Matplotlib                               MonteCarlo.cpp
  CSV Batch I/O                            ActuarialMath.cpp
                                           LossTriangle.cpp
                                           ExperienceRating.cpp
                                           ExposureRating.cpp
                                           RateAnalyzer.cpp
                                           ExcelExporter.cpp (OpenXLSX)
```

Built using `scikit-build-core` + `CMake`. Published via GitHub Actions CI/CD with PyPI Trusted Publishers (OIDC).

## License
MIT — Free for personal and commercial use.

## Links
- **GitHub**: [github.com/slimboi34/RiskPY](https://github.com/slimboi34/RiskPY)
- **PyPI**: [pypi.org/project/open-riskpy](https://pypi.org/project/open-riskpy)
