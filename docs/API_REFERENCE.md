# RiskPY API Reference

Complete documentation for every class and method available in the `riskpy` package.

```bash
pip install open-riskpy
```

---

## Table of Contents

- [FactorModel](#factormodel) — Declarative multiplicative rating engine
- [MonteCarloSimulator](#montecarlosimulator) — Stochastic simulation engine
- [ActuarialMath](#actuarialmath) — Core actuarial formulas
- [UnderwritingApp](#underwritingapp) — Interactive GUI application
- [RiskEngine](#riskengine) — Low-level C++ calculation orchestrator
- [ExcelExporter](#excelexporter) — Binary Excel output engine

---

## FactorModel

A declarative, multiplicative rating engine. Define pricing rules as factors — the C++ engine applies them instantly across any number of policies.

### Constructor

```python
from riskpy import FactorModel

model = FactorModel(initial_base_rate=1000.0)
```

| Parameter | Type | Description |
|---|---|---|
| `initial_base_rate` | `float` | The starting premium before any factors are applied. Default: `0.0` |

### Methods

#### `set_base_rate(rate)`
Updates the base rate after construction.

```python
model.set_base_rate(1500.0)
```

| Parameter | Type | Description |
|---|---|---|
| `rate` | `float` | New base premium amount |

---

#### `add_multiplier(field_name, exact_value, multiplier)`
Adds a categorical factor rule. When the input field exactly matches the value, the premium is multiplied by the factor.

```python
model.add_multiplier("state", "CA", 2.0)   # California = 2x
model.add_multiplier("state", "FL", 3.0)   # Florida = 3x (hurricane risk)
model.add_multiplier("vehicle_type", "Sports Car", 2.5)
```

| Parameter | Type | Description |
|---|---|---|
| `field_name` | `str` | The name of the input field to match against |
| `exact_value` | `str` | The exact string value that triggers this multiplier |
| `multiplier` | `float` | The factor to multiply the premium by (e.g., `2.0` = double) |

---

#### `add_numeric_band_multiplier(field_name, min_val, max_val, multiplier)`
Adds a continuous band factor. When the numeric input falls within `[min_val, max_val]`, the multiplier is applied.

```python
model.add_numeric_band_multiplier("driver_age", 16, 25, 2.0)   # Young driver surcharge
model.add_numeric_band_multiplier("driver_age", 26, 65, 1.0)   # Standard rate
model.add_numeric_band_multiplier("driver_age", 66, 99, 1.3)   # Senior surcharge
```

| Parameter | Type | Description |
|---|---|---|
| `field_name` | `str` | The name of the numeric input field |
| `min_val` | `float` | Lower bound of the band (inclusive) |
| `max_val` | `float` | Upper bound of the band (inclusive) |
| `multiplier` | `float` | The factor to apply when the value falls in this band |

---

#### `calculate(inputs)`
Evaluates all rules against a dictionary of inputs and returns the final premium.

```python
premium = model.calculate({"state": "CA", "driver_age": 22.0})
# Result: 1000.0 × 2.0 (CA) × 2.0 (age 16-25) = $4,000.00
```

| Parameter | Type | Description |
|---|---|---|
| `inputs` | `dict[str, str \| float]` | A dictionary mapping field names to their values |

**Returns**: `float` — The calculated premium after all matching factors are applied.

---

## MonteCarloSimulator

High-performance stochastic simulation engine running natively in C++. All methods are static — no instantiation required.

### Methods

#### `simulate_aggregate_loss(trials, expected_frequency, expected_severity_mu, severity_sigma)`
Simulates aggregate insurance losses for Property & Casualty portfolios. Each trial generates a random number of claims (Poisson distributed) with random severity amounts (Lognormal distributed), then sums them.

```python
from riskpy import MonteCarloSimulator

results = MonteCarloSimulator.simulate_aggregate_loss(
    trials=100000,
    expected_frequency=5.0,
    expected_severity_mu=10.0,
    severity_sigma=1.5
)

# results is a list of 100,000 floats, each representing one simulated year's total losses
import numpy as np
var_99 = np.percentile(results, 99)
print(f"99% Value-at-Risk: ${var_99:,.0f}")
```

| Parameter | Type | Description |
|---|---|---|
| `trials` | `int` | Number of simulation iterations (e.g., `100000`) |
| `expected_frequency` | `float` | Average number of claims per period — the λ parameter for Poisson distribution |
| `expected_severity_mu` | `float` | Mean of the log of severity — the μ parameter for Lognormal distribution |
| `severity_sigma` | `float` | Standard deviation of the log of severity — the σ parameter for Lognormal distribution |

**Returns**: `list[float]` — Array of aggregated loss amounts, one per trial.

**Use cases**: Catastrophe modeling, reinsurance pricing, capital requirements (Solvency II, IFRS 17), determining reserves.

---

#### `simulate_life_portfolio(trials, policy_count, base_mortality_rate, shock_volatility, death_benefit)`
Simulates total mortality claims for a life insurance portfolio. Each trial applies a stochastic shock to the base mortality rate, then determines how many policyholders die and calculates total claims.

```python
results = MonteCarloSimulator.simulate_life_portfolio(
    trials=50000,
    policy_count=10000,
    base_mortality_rate=0.001,
    shock_volatility=0.2,
    death_benefit=500000.0
)

# results is a list of 50,000 total claim amounts
expected_claims = np.mean(results)
print(f"Expected annual claims: ${expected_claims:,.0f}")
```

| Parameter | Type | Description |
|---|---|---|
| `trials` | `int` | Number of simulation iterations |
| `policy_count` | `int` | Number of policies in the portfolio |
| `base_mortality_rate` | `float` | Base probability of death per policyholder per period (e.g., `0.001` = 0.1%) |
| `shock_volatility` | `float` | Standard deviation of the multiplicative mortality shock (Normal distribution centered at 1.0) |
| `death_benefit` | `float` | Payout amount per death claim |

**Returns**: `list[float]` — Array of total portfolio claim amounts, one per trial.

**Use cases**: Life portfolio risk assessment, pandemic scenario testing, mortality assumption validation.

---

## ActuarialMath

Core actuarial formulas implemented in C++ for maximum precision and speed. All methods are static.

### Methods

#### `present_value(rate, periods, payment)`
Calculates the present value of an ordinary annuity — the current worth of a series of future payments.

```python
from riskpy import ActuarialMath

# What is $50,000/year for 20 years worth today at 5% discount?
pv = ActuarialMath.present_value(rate=0.05, periods=20, payment=50000)
print(f"Present value: ${pv:,.2f}")  # $623,110.52
```

| Parameter | Type | Description |
|---|---|---|
| `rate` | `float` | Discount rate per period (e.g., `0.05` for 5%) |
| `periods` | `int` | Number of payment periods |
| `payment` | `float` | Payment amount per period |

**Returns**: `float` — Present value of the annuity.

---

#### `future_value(rate, periods, payment)`
Calculates the future value of an ordinary annuity — what a series of payments will grow to.

```python
fv = ActuarialMath.future_value(rate=0.07, periods=30, payment=10000)
print(f"Future value: ${fv:,.2f}")
```

| Parameter | Type | Description |
|---|---|---|
| `rate` | `float` | Interest rate per period |
| `periods` | `int` | Number of payment periods |
| `payment` | `float` | Payment amount per period |

**Returns**: `float` — Future value of the annuity.

---

#### `calculate_loss_ratio(incurred_losses, earned_premium)`
Calculates the loss ratio — a critical P&C insurance KPI.

```python
lr = ActuarialMath.calculate_loss_ratio(incurred_losses=750000, earned_premium=1000000)
print(f"Loss ratio: {lr:.1%}")  # 75.0%
```

| Parameter | Type | Description |
|---|---|---|
| `incurred_losses` | `float` | Total incurred losses |
| `earned_premium` | `float` | Total earned premium |

**Returns**: `float` — Loss ratio (e.g., `0.75` for 75%).

---

#### `lookup_mortality_rate(age)`
Looks up the mortality rate (qx) for a given age using a built-in CSO 2001 table.

```python
qx = ActuarialMath.lookup_mortality_rate(age=65)
print(f"Mortality rate at age 65: {qx:.4f}")
```

| Parameter | Type | Description |
|---|---|---|
| `age` | `int` | Age of the insured (integer) |

**Returns**: `float` — Probability of death within one year (qx).

---

## UnderwritingApp

The high-level Python class that provides a complete GUI application. It wraps the C++ engine, Excel exporter, and Monte Carlo simulator into an interactive Tkinter desktop app.

### Constructor

```python
from riskpy import UnderwritingApp

app = UnderwritingApp(title="My Pricing Tool", excel_template="template.xlsx")
```

| Parameter | Type | Description |
|---|---|---|
| `title` | `str` | Window title for the GUI. Default: `"Actuarial Underwriter"` |
| `excel_template` | `str` | Path to Excel template file. Default: `"corporate_layout.xlsx"` |

### Methods

#### `add_field(name, label, excel_col, choices=None)`
Adds an input field to the GUI form and maps it to an Excel column.

```python
app.add_field("state", "Location State", "A", choices=["NY", "CA", "FL"])
app.add_field("age", "Driver Age", "B")  # free-text numeric input
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Internal field identifier |
| `label` | `str` | Human-readable label shown in the GUI |
| `excel_col` | `str` | Excel column letter for export (e.g., `"A"`, `"B"`) |
| `choices` | `list[str]` or `None` | If provided, renders a dropdown. If `None`, renders a text input |

---

#### `set_factor_model(factor_model)`
Connects a `FactorModel` instance as the calculation engine.

```python
model = FactorModel(initial_base_rate=500.0)
model.add_multiplier("state", "FL", 3.0)
app.set_factor_model(model)
```

| Parameter | Type | Description |
|---|---|---|
| `factor_model` | `FactorModel` | A configured `FactorModel` instance |

---

#### `set_logic(logic_func)`
Sets a custom Python function as the calculation logic (alternative to `FactorModel`).

```python
def my_pricing(inputs):
    base = 1000
    if inputs.get("state") == "CA":
        base *= 2
    return base

app.set_logic(my_pricing)
```

| Parameter | Type | Description |
|---|---|---|
| `logic_func` | `callable` | A function that accepts `dict` inputs and returns a `float` premium |

---

#### `calculate_headless(inputs)`
Runs a single calculation without opening the GUI.

```python
premium = app.calculate_headless({"state": "CA", "age": 22.0})
print(f"${premium:,.2f}")
```

| Parameter | Type | Description |
|---|---|---|
| `inputs` | `dict` | Field name to value mapping |

**Returns**: `float` — Calculated premium.

---

#### `calculate_batch(csv_filepath, output_filename)`
Processes an entire CSV file of policy inputs and exports results to Excel.

```python
total, count = app.calculate_batch("policies.csv", "results.xlsx")
print(f"Processed {count} policies. Total book premium: ${total:,.2f}")
```

| Parameter | Type | Description |
|---|---|---|
| `csv_filepath` | `str` | Path to the CSV input file |
| `output_filename` | `str` | Path for the Excel output file. Default: `"batch_policy_quotes.xlsx"` |

**Returns**: `tuple[float, int]` — (total premium, number of records processed).

---

#### `run(export_filename)`
Launches the interactive Tkinter GUI with two tabs: Pricing & Rating, and Monte Carlo Predictor.

```python
app.run()
```

| Parameter | Type | Description |
|---|---|---|
| `export_filename` | `str` | Default filename for single-quote Excel exports. Default: `"policy_quote.xlsx"` |

---

## RiskEngine

Low-level C++ calculation orchestrator. Most users should use `UnderwritingApp` instead, which wraps this class. Exposed for advanced use cases.

```python
from riskpy import RiskEngine, Field

engine = RiskEngine()
engine.add_field(Field(name="age", label="Age", type="float"))
engine.set_math_logic(my_function)
result = engine.execute({"age": 25.0})
```

### Methods

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `add_field(field)` | `Field` | `None` | Registers an input field |
| `get_fields()` | — | `list[Field]` | Returns all registered fields |
| `set_math_logic(func)` | `callable` | `None` | Sets the pricing function |
| `attach_exporter(exporter)` | `ExcelExporter` | `None` | Connects Excel output |
| `execute(inputs)` | `dict` | `float` | Runs calculation |
| `export_to_excel(filename)` | `str` | `None` | Writes results to Excel |

---

## ExcelExporter

Writes calculation results to binary `.xlsx` files using the C++ OpenXLSX library (bypasses Python entirely for speed).

```python
from riskpy import ExcelExporter

exporter = ExcelExporter(template="my_template.xlsx")
exporter.map_column("A", "state", "State")
exporter.map_column("B", "age", "Driver Age")
```

### Methods

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `map_column(col, field, label)` | `str`, `str`, `str` | `None` | Maps a field to an Excel column |
