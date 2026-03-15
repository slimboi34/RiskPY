# RiskPY

A hyper-fast, C++ powered pricing framework built explicitly for actuaries across all lines of insurance (P&C, Life, Health).

## The Painkiller
Currently, actuaries are forced to write complex, slow `if/else` ladders in Python or rely on clunky Excel spreadsheets to process rating models and massive stochastic simulations. `RiskPY` solves this by providing a declarative, high-performance C++ backend combined with a gorgeous built-in Desktop UI.

You write clean, readable pricing rules in Python. We execute them instantly in C++.

## Features
- **Native FactorModel Engine**: A declarative C++ rating engine matching top P&C insurance pricing strategies.
- **Built-in ActuarialMath**: Core life/annuity computations exposed directly from C++ to Python.
- **Bulk CSV Batching**: Crunch 10,000+ policy quotes simultaneously without Python GIL bottlenecks.
- **Monte Carlo Predictive Modeling**: Simulate millions of Stochastic Tail Risk scenarios natively in C++ (Lognormal Severities for P&C, Systemic Mortality Shocks for Life) instantly generating predictive Histograms in Tkinter.

## Documentation
- **[API Reference](docs/API_REFERENCE.md)**: Complete method-by-method documentation with parameters, return types, and code examples for all 6 classes.
- **[Tutorials](docs/TUTORIALS.md)**: Step-by-step guides for building auto insurance raters, catastrophe models, life portfolio stress tests, and batch CSV processors.
- **[Architecture](docs/ARCHITECTURE.md)**: Deep technical dive into the C++/Python bridge, pybind11 internals, algorithm implementations, and CI/CD pipeline.
- **[How It Works](docs/HOW_IT_WORKS.md)**: High-level overview of the hybrid architecture.
- **[Use Cases](docs/USE_CASES.md)**: Real-world insurance pricing scenarios with code.
- **[Publishing Guide](docs/PUBLISHING.md)**: How to contribute and distribute.

## Examples
Complete, runnable example scripts are included in the [`examples/`](examples/) directory:
- **[Auto Insurance Rater](examples/auto_insurance_rater.py)** — Full P&C rating engine with territorial, age, vehicle, and claims factors
- **[Life Annuity Pricing](examples/life_annuity_pricing.py)** — Present value, future value, mortality lookups, and portfolio simulation
- **[Catastrophe Model](examples/catastrophe_model.py)** — Hurricane risk modeling with 200K+ stochastic scenarios and reinsurance analysis
- **[Batch Processing](examples/batch_processing.py)** — Process 500 policy submissions from CSV and export to Excel

## Installation
```bash
pip install open-riskpy
```

## Quick Start

### 1. Declarative Pricing (3 lines of config)
```python
from riskpy import FactorModel

model = FactorModel(initial_base_rate=1000.0)
model.add_multiplier("state", "FL", 3.0)               # Florida hurricane surcharge
model.add_numeric_band_multiplier("age", 16, 25, 2.0)   # Young driver surcharge

premium = model.calculate({"state": "FL", "age": 19.0})
# Result: $1,000 × 3.0 × 2.0 = $6,000.00
```

### 2. Monte Carlo Simulation (1 line of C++)
```python
from riskpy import MonteCarloSimulator
import numpy as np

# 100K stochastic scenarios in milliseconds
losses = MonteCarloSimulator.simulate_aggregate_loss(
    trials=100000, expected_frequency=5.0,
    expected_severity_mu=10.0, severity_sigma=1.5
)
print(f"99% VaR: ${np.percentile(losses, 99):,.0f}")
```

### 3. Core Actuarial Math
```python
from riskpy import ActuarialMath

pv = ActuarialMath.present_value(rate=0.05, periods=20, payment=50000)
lr = ActuarialMath.calculate_loss_ratio(750000, 1000000)
qx = ActuarialMath.lookup_mortality_rate(age=65)
```

### 4. Full GUI Application
```python
from riskpy import UnderwritingApp, FactorModel

app = UnderwritingApp(title="Actuarial Pricing Tool")
app.add_field("state", "Location State", "A", choices=["NY", "CA", "FL"])
app.add_field("age", "Client Age", "B")

model = FactorModel(initial_base_rate=500.0)
model.add_multiplier("state", "CA", 2.0)
app.set_factor_model(model)

app.run()  # Launches desktop GUI with Pricing + Monte Carlo tabs
```

### 5. Headless Batch Processing
```python
total_premium, count = app.calculate_batch("policies.csv", "output.xlsx")
print(f"Processed {count} policies. Total: ${total_premium:,.2f}")
```

## License
MIT — Free for personal and commercial use.
