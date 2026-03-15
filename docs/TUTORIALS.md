# Tutorials

Step-by-step guides for building real actuarial applications with RiskPY.

---

## Tutorial 1: Building an Auto Insurance Rater from Scratch

**Goal**: Create a complete auto insurance pricing tool with a GUI in under 30 lines of code.

**What You'll Learn**: `FactorModel`, `UnderwritingApp`, `add_field`, `add_multiplier`, `add_numeric_band_multiplier`

### Step 1: Install RiskPY

```bash
pip install open-riskpy
```

### Step 2: Import the Framework

```python
from riskpy import UnderwritingApp, FactorModel
```

These are the two main classes you need. `FactorModel` handles the math. `UnderwritingApp` handles the GUI and Excel export.

### Step 3: Define Your Base Rate

Every insurance product starts with a **base rate** — the average expected cost before any risk adjustments.

```python
model = FactorModel(initial_base_rate=800.0)
```

In practice, this number comes from your actuarial loss cost analysis. For this tutorial, we'll use $800.

### Step 4: Add Categorical Rating Factors

**Categorical factors** apply a multiplier when a field matches an exact string value. These are your territorial, class, and experience rating factors.

```python
# Territorial rating — each state has different risk profiles
model.add_multiplier("state", "NY", 1.8)
model.add_multiplier("state", "CA", 2.0)
model.add_multiplier("state", "FL", 2.5)

# Vehicle class rating
model.add_multiplier("vehicle_type", "Sedan", 1.0)
model.add_multiplier("vehicle_type", "Sports Car", 2.0)
```

**How it works internally**: When `calculate()` is called, the C++ engine iterates through all registered rules. If the input dictionary contains `{"state": "FL"}`, it finds the matching rule and multiplies the premium by 2.5. If there's no match, the factor defaults to 1.0 (no change).

### Step 5: Add Continuous Band Factors

**Band factors** apply when a numeric value falls within a range. This is essential for age rating, mileage bands, property values, etc.

```python
model.add_numeric_band_multiplier("driver_age", 16, 25, 2.0)   # Young driver surcharge
model.add_numeric_band_multiplier("driver_age", 26, 65, 1.0)   # Standard rate
model.add_numeric_band_multiplier("driver_age", 66, 99, 1.3)   # Senior adjustment
```

**How it works internally**: The C++ engine checks if the numeric input falls within `[min_val, max_val]` inclusive. If a 22-year-old is quoted, `16 <= 22 <= 25` is true, so the 2.0x factor is applied.

### Step 6: Build the GUI

```python
app = UnderwritingApp(title="Auto Insurance Rater")
app.add_field("state", "State", "A", choices=["NY", "CA", "FL"])
app.add_field("driver_age", "Driver Age", "B")
app.add_field("vehicle_type", "Vehicle Type", "C", choices=["Sedan", "Sports Car"])
app.set_factor_model(model)
app.run()
```

- Fields with `choices` render as **dropdowns** in the GUI
- Fields without `choices` render as **text inputs** (for numbers)
- The Excel column letter (`"A"`, `"B"`, etc.) controls where data appears in the exported spreadsheet

### Step 7: Run It

```bash
python my_rater.py
```

A desktop window opens with two tabs:
1. **Pricing & Rating** — Fill in the form, click "Calculate Profile", get a premium, and export to Excel
2. **Monte Carlo Predictor** — Run stochastic simulations on your portfolio

**Full working example**: See [`examples/auto_insurance_rater.py`](../examples/auto_insurance_rater.py)

---

## Tutorial 2: Modelling Catastrophe Tail Risk

**Goal**: Simulate 200,000 hurricane seasons to determine the 1-in-200 year loss for capital requirements.

**What You'll Learn**: `MonteCarloSimulator.simulate_aggregate_loss`, Poisson/Lognormal distributions, Value-at-Risk

### The Actuarial Problem

Regulators (Solvency II, IFRS 17) require insurers to hold sufficient capital to survive a 1-in-200 year catastrophe. To calculate this, you need to simulate hundreds of thousands of possible loss scenarios.

### Step 1: Define Your Assumptions

```python
from riskpy import MonteCarloSimulator
import numpy as np
```

The two key assumptions for aggregate loss modelling:
- **Frequency**: How many events happen? → Modelled by a **Poisson distribution** with parameter λ (expected count)
- **Severity**: How big is each event? → Modelled by a **Lognormal distribution** with parameters μ (log-mean) and σ (log-standard-deviation)

### Step 2: Run the Simulation

```python
results = MonteCarloSimulator.simulate_aggregate_loss(
    trials=200000,         # Number of simulated years
    expected_frequency=5.0, # Average 5 hurricane landfalls per year
    expected_severity_mu=17.7, # ln($50M average loss) ≈ 17.7
    severity_sigma=1.2     # High volatility in loss amounts
)
```

**What happens inside C++**:
1. For each of the 200,000 trials, a random claim count `N` is drawn from `Poisson(λ=5.0)`
2. For each of the `N` claims, a random severity is drawn from `Lognormal(μ=17.7, σ=1.2)`
3. The claims are summed to get the total aggregate loss for that trial
4. The array of 200,000 aggregate totals is returned to Python

This entire process takes **less than 1 second** in C++. In pure Python, it would take minutes.

### Step 3: Analyse the Results

```python
print(f"Expected annual loss:  ${np.mean(results):,.0f}")
print(f"95% VaR:               ${np.percentile(results, 95):,.0f}")
print(f"99% VaR:               ${np.percentile(results, 99):,.0f}")
print(f"99.5% VaR (1-in-200):  ${np.percentile(results, 99.5):,.0f}")
```

The **99.5th percentile** is your 1-in-200 year loss — the amount regulators require you to hold as capital.

**Full working example**: See [`examples/catastrophe_model.py`](../examples/catastrophe_model.py)

---

## Tutorial 3: Life Portfolio Mortality Stress Testing

**Goal**: Determine how much capital a life insurer needs to survive a pandemic-level mortality shock.

**What You'll Learn**: `MonteCarloSimulator.simulate_life_portfolio`, mortality shocks, portfolio risk

### The Model

```python
results = MonteCarloSimulator.simulate_life_portfolio(
    trials=50000,
    policy_count=10000,          # 10,000 life policies
    base_mortality_rate=0.002,   # 0.2% base annual mortality (qx)
    shock_volatility=0.3,        # 30% volatility in mortality shock
    death_benefit=500000.0       # $500K payout per death
)
```

**What happens inside C++**:
1. For each trial, a **mortality shock factor** is drawn from `Normal(mean=1.0, std=0.3)`
2. The effective mortality rate = `base_rate × shock_factor` (clamped to [0, 1])
3. For each of the 10,000 policies, a random draw determines if the policyholder dies
4. Total claims = number of deaths × death benefit

A shock factor of 1.5 represents a 50% increase in mortality (pandemic scenario). A shock factor of 0.8 represents a mild year.

**Full working example**: See [`examples/life_annuity_pricing.py`](../examples/life_annuity_pricing.py)

---

## Tutorial 4: Processing a Broker Submission Book

**Goal**: Rate 500 policy submissions from a CSV file and export results to Excel for the underwriting team.

**What You'll Learn**: `calculate_batch`, CSV processing, Excel export

### Step 1: Prepare Your CSV

Your CSV should have headers matching your field names:

```csv
state,driver_age,vehicle_type,claims_history
NY,34,Sedan,Clean
FL,19,Sports Car,2+ Claims
CA,45,SUV,1 Claim
```

### Step 2: Process the Batch

```python
total_premium, count = app.calculate_batch("submissions.csv", "rated_book.xlsx")
print(f"Processed {count} policies. Total: ${total_premium:,.2f}")
```

The C++ engine reads each row, applies all `FactorModel` rules, calculates the premium, and writes the result directly to a binary Excel file using the OpenXLSX C++ library — completely bypassing Python's GIL.

**Full working example**: See [`examples/batch_processing.py`](../examples/batch_processing.py)
