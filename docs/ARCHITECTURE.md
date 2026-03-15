# Architecture

A deep-dive into how RiskPY bridges Python and C++ to deliver actuarial computations at native speed.

---

## System Overview

```
┌─────────────────────────────────────────────────┐
│                  USER'S PYTHON CODE              │
│  from riskpy import FactorModel, MonteCarloSim  │
└──────────────────────┬──────────────────────────┘
                       │ Python function calls
                       ▼
┌─────────────────────────────────────────────────┐
│              riskpy/ (Python Package)            │
│  ┌──────────────────────────────────────────┐   │
│  │  __init__.py   — Public API surface       │   │
│  │  app.py        — Tkinter GUI + Matplotlib │   │
│  └──────────────────────────────────────────┘   │
│                       │                          │
│           ┌───────────┴────────────┐             │
│           │  cpp_underwriter.so    │             │
│           │  (Compiled C++ Binary) │             │
│           └───────────┬────────────┘             │
└───────────────────────┼─────────────────────────┘
                        │ Direct C++ method calls
                        ▼
┌─────────────────────────────────────────────────┐
│              C++ ENGINE (src/)                   │
│                                                  │
│  FactorModel.cpp    — Multiplicative rating      │
│  MonteCarlo.cpp     — Stochastic simulation      │
│  ActuarialMath.cpp  — Financial mathematics      │
│  RiskEngine.cpp     — Calculation orchestrator   │
│  ExcelExporter.cpp  — Binary Excel output        │
│  bindings.cpp       — pybind11 bridge            │
│                                                  │
│  Dependencies:                                   │
│  ├── pybind11 (C++/Python bridge)                │
│  ├── OpenXLSX (Excel I/O)                        │
│  ├── <random>  (Poisson, Lognormal, Normal)      │
│  └── <cmath>   (PV, FV formulas)                 │
└─────────────────────────────────────────────────┘
```

---

## The Build Pipeline

RiskPY uses `scikit-build-core` to compile C++ code during `pip install`. Here's the full chain:

### 1. User runs `pip install open-riskpy`
### 2. pip reads `pyproject.toml`

```toml
[build-system]
requires = ["scikit-build-core>=0.5.0", "pybind11>=2.11.0"]
build-backend = "scikit_build_core.build"
```

### 3. scikit-build-core invokes CMake

CMake reads `CMakeLists.txt`, which:
- Fetches `pybind11` from GitHub (for Python/C++ bridging)
- Fetches `OpenXLSX` from GitHub (for Excel output)
- Compiles all `.cpp` files in `src/` into a shared library

### 4. The compiled binary is installed

The resulting `cpp_underwriter.cpython-311-darwin.so` (macOS) or `.pyd` (Windows) is placed inside the `riskpy/` package directory. Python imports it as a regular module.

---

## How pybind11 Works

pybind11 creates a bridge between C++ classes and Python objects. In `bindings.cpp`:

```cpp
py::class_<FactorModel>(m, "FactorModel")
    .def(py::init<double>(), py::arg("initial_base_rate") = 0.0)
    .def("add_multiplier", &FactorModel::add_multiplier,
         py::arg("field_name"), py::arg("exact_value"), py::arg("multiplier"))
    .def("calculate", &FactorModel::calculate, py::arg("inputs"));
```

This tells pybind11:
1. Create a Python class called `FactorModel`
2. Its constructor takes a `double` (maps to Python `float`)
3. It has a method `add_multiplier` taking 3 arguments
4. It has a method `calculate` that accepts a Python `dict` and returns a `float`

When Python calls `model.calculate({"state": "FL"})`, pybind11 automatically converts the Python `dict` into a C++ `std::map<std::string, std::variant<double, std::string>>` and passes it to the native C++ function.

---

## How the FactorModel Engine Works

The `FactorModel` uses a simple but powerful multiplicative architecture:

```
Final Premium = Base Rate × Factor₁ × Factor₂ × Factor₃ × ...
```

### Internal Data Structures

```cpp
struct ExactMatchRule {
    std::string field_name;    // "state"
    std::string exact_value;   // "FL"
    double multiplier;         // 2.5
};

struct NumericBandRule {
    std::string field_name;    // "driver_age"
    double min_val;            // 16
    double max_val;            // 25
    double multiplier;         // 2.0
};
```

### Calculation Algorithm

```
1. Start with base_rate
2. For each ExactMatchRule:
   - If inputs[rule.field_name] == rule.exact_value:
       premium *= rule.multiplier
3. For each NumericBandRule:
   - If rule.min_val <= inputs[rule.field_name] <= rule.max_val:
       premium *= rule.multiplier
4. Return premium
```

This runs entirely in C++ with zero Python overhead. Processing 100,000 policies takes milliseconds.

---

## How the Monte Carlo Engine Works

### Aggregate Loss Simulation (P&C)

The aggregate loss model combines two distributions:

**Frequency** (how many claims): `N ~ Poisson(λ)`
- λ = expected number of claims per period
- A Poisson draw might return 0, 3, 7, or 12 claims

**Severity** (how big each claim): `X ~ Lognormal(μ, σ)`
- μ = mean of the log of losses
- σ = standard deviation of the log of losses

**Aggregate Loss** for one trial:
```
L = X₁ + X₂ + ... + Xₙ  (where N is drawn from Poisson)
```

### C++ Implementation

```cpp
std::poisson_distribution<int> freq_dist(expected_frequency);
std::lognormal_distribution<double> sev_dist(expected_severity_mu, severity_sigma);

for (int trial = 0; trial < trials; trial++) {
    int num_claims = freq_dist(generator);
    double total_loss = 0.0;
    for (int j = 0; j < num_claims; j++) {
        total_loss += sev_dist(generator);
    }
    results[trial] = total_loss;
}
```

The C++ `<random>` library provides hardware-optimised Mersenne Twister generators, making 1,000,000 trials complete in under 1 second.

### Life Portfolio Simulation

```cpp
std::normal_distribution<double> shock_dist(1.0, shock_volatility);

for (int trial = 0; trial < trials; trial++) {
    double shock = shock_dist(generator);           // e.g., 1.3 = 30% mortality increase
    double effective_qx = base_qx * std::max(0.0, shock);
    int deaths = 0;
    for (int p = 0; p < policy_count; p++) {
        if (uniform_dist(generator) < effective_qx)
            deaths++;
    }
    results[trial] = deaths * death_benefit;
}
```

---

## CI/CD Pipeline

RiskPY uses GitHub Actions for automated publishing:

```
Developer pushes code → GitHub detects release tag
    → Ubuntu runner spins up
    → Installs cmake + g++ + Python
    → scikit-build-core compiles C++ into source distribution
    → pypa/gh-action-pypi-publish uploads to PyPI
    → Package is live worldwide in ~60 seconds
```

The workflow file (`.github/workflows/publish.yml`) uses **Trusted Publishers** (OIDC) to authenticate with PyPI — no passwords or API tokens stored anywhere.

---

## File Structure

```
underwriting_python_framework/
├── .github/workflows/
│   └── publish.yml          # CI/CD automation
├── docs/
│   ├── API_REFERENCE.md     # Method-by-method documentation
│   ├── ARCHITECTURE.md      # This file
│   ├── HOW_IT_WORKS.md      # High-level overview
│   ├── PUBLISHING.md        # Distribution guide
│   ├── TUTORIALS.md         # Step-by-step guides
│   └── USE_CASES.md         # Actuarial scenarios
├── examples/
│   ├── auto_insurance_rater.py
│   ├── batch_processing.py
│   ├── catastrophe_model.py
│   └── life_annuity_pricing.py
├── riskpy/
│   ├── __init__.py           # Public API exports
│   └── app.py                # Tkinter GUI + Matplotlib
├── src/
│   ├── ActuarialMath.cpp/hpp # Financial math
│   ├── ExcelExporter.cpp/hpp # Binary Excel I/O
│   ├── FactorModel.cpp/hpp   # Rating engine
│   ├── Field.hpp             # Data structures
│   ├── MonteCarlo.cpp/hpp    # Stochastic simulation
│   ├── RiskEngine.cpp/hpp    # Orchestrator
│   └── bindings.cpp          # pybind11 bridge
├── CMakeLists.txt            # C++ build config
├── LICENSE                   # MIT
├── README.md                 # Project overview (also PyPI page)
└── pyproject.toml            # Python package config
```
