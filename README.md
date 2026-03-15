# RiskPY

A hyper-fast, C++ powered pricing framework built explicitly for actuaries across all lines of insurance (P&C, Life, Health).

## The Painkiller
Currently, actuaries are forced to write complex, slow `if/else` ladders in Python or rely on clunky Excel spreadsheets to process rating models and massive stochastic simulations. `RiskPY` solves this by providing a declarative, high-performance C++ backend combined with a gorgeous built-in Desktop UI.

You write clean, readable pricing rules in Python. We execute them instantly in C++.

## Features
- **Native FactorModel Engine**: A declarative C++ rating engine matching top P&C insurance pricing stategies.
- **Built-in ActuarialMath**: Core life/annuity computations exposed directly from C++ to python.
- **Bulk CSV Batching**: Crunch 10,000+ policy quotes simultaneously without Python GIL bottlenecks.
- **Monte Carlo Predictive Modeling**: Simulate millions of Stochastic Tail Risk scenarios natively in C++ (Lognormal Severities for P&C, Systemic Mortality Shocks for Life) instantly generating predictive Histograms in Tkinter.

## Documentation
- **[How It Works](docs/HOW_IT_WORKS.md)**: Explore the dual C++ to Python architecture allowing zero-cost rule evaluation.
- **[Actuarial Use Cases](docs/USE_CASES.md)**: Explore example snippets building P&C Multiplicative Raters, Lognormal Monte Carlo Tail Sims, and Core Life Annuity Present Value tables.
- **[Publishing & Accessibility](docs/PUBLISHING.md)**: Discover how to upload this module to GitHub natively and distribute it across the world via PyPI `twine`.

## Installation
```bash
pip install riskpy
```

## Quick Start
```python
from riskpy import UnderwritingApp, FactorModel

app = UnderwritingApp(title="Actuarial Pricing Tool")

# 1. Declare User Form Fields
app.add_field("state", "Location State", "A", choices=["NY", "CA", "FL"])
app.add_field("age", "Client Age", "B")

# 2. Build the Multiplicative Rating Engine
model = FactorModel(initial_base_rate=500.0)
model.add_multiplier("state", "CA", 2.0)  
model.add_multiplier("state", "FL", 3.0)  
model.add_numeric_band_multiplier("age", 18, 25, 1.4) 

app.set_factor_model(model)

# 3. Launch the Dual-Pane GUI (Pricing forms & Interactive Monte Carlo Simulations)
app.run()
```
