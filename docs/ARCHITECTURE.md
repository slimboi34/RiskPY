# Architecture

A deep-dive into how RiskPY bridges Python and C++ to deliver actuarial computations at native speed, and where the pure-Python modelling layers sit.

---

## System Overview

```
your code         from riskpy import FactorModel, mc, life, reserving, ...
   │
   ▼
riskpy/           the Python package
   ├── __init__.py      public API; loads the layers below lazily, on first use
   ├── mc.py            generic Monte Carlo
   ├── viz.py           charts
   ├── quant.py         option pricing, paths, portfolio risk
   ├── life.py          life contingencies
   ├── reserving.py     claims reserving
   ├── rates.py         yield curves, bonds, short-rate models
   ├── credit.py        credit risk
   ├── _special.py      special functions and root finders, without SciPy
   ├── verify.py        the verification suite
   ├── app.py           Tkinter GUI + Matplotlib (UnderwritingApp)
   └── cpp_underwriter  the compiled extension (.so / .pyd)
          │
          │ pybind11
          ▼
src/              C++17
   ├── FactorModel        multiplicative rating
   ├── MonteCarlo         stochastic simulation (MonteCarloSimulator)
   ├── ActuarialMath      annuity PV/FV, loss ratio, an illustrative mortality lookup
   ├── LossTriangle       chain-ladder reserving
   ├── ExperienceRating   credibility and experience modification
   ├── ExposureRating     increased limits factors, layer premiums, burning cost
   ├── RateAnalyzer       on-level premiums, trend, indicated rate change
   ├── FourierMath        FFT and the compound Poisson PMF (FourierTransform)
   ├── RiskEngine         calculation orchestrator behind UnderwritingApp
   ├── ExcelExporter      binary .xlsx output via OpenXLSX
   └── bindings.cpp       the pybind11 bridge
```

The modelling layers (`mc` through `credit`) are ordinary Python on NumPy and do
not call into C++. The compiled core has no Python dependencies at all, which is
why `import riskpy` works on a machine with nothing but the wheel installed.

---

## The Build Pipeline

RiskPY uses `scikit-build-core` to compile C++ code during `pip install`. Most
users never see this: pip picks a pre-built wheel (CPython 3.10–3.14 on Linux,
macOS and Windows) whenever one matches. Here's the chain for a source build:

### 1. User runs `pip install open-riskpy`
### 2. pip reads `pyproject.toml`

```toml
[build-system]
requires = ["scikit-build-core>=0.10.0", "pybind11>=2.12.0"]
build-backend = "scikit_build_core.build"
```

### 3. scikit-build-core invokes CMake

CMake reads `CMakeLists.txt`, which:
- Uses the `pybind11` installed from `build-system.requires`, and only fetches it from GitHub if that is missing
- Fetches `OpenXLSX` v0.5.1 from GitHub (for Excel output)
- Compiles all `.cpp` files in `src/` into one extension module, `cpp_underwriter`

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
    std::string exact_value;   // "FL"
    double multiplier;         // 2.5
};

struct NumericBandRule {
    double min_val;            // 16
    double max_val;            // 25
    double multiplier;         // 2.0
};

// Rules keyed by field name, so a quote only touches the rules for its own fields
std::unordered_map<std::string, std::vector<ExactMatchRule>> exact_rules;
std::unordered_map<std::string, std::vector<NumericBandRule>> band_rules;
```

### Calculation Algorithm

```
1. Start with base_rate
2. For each (field, value) in the inputs:
   - If value is text, for each exact rule on that field:
       if value == rule.exact_value: premium *= rule.multiplier
   - If value is a number, for each band rule on that field:
       if rule.min_val <= value <= rule.max_val: premium *= rule.multiplier
3. Return premium
```

Rule evaluation is pure C++. Each call from Python still pays pybind11's
conversion of the input `dict` into the `std::map`, so batch work is fastest
when the loop itself stays small.

---

## How the Monte Carlo Engine Works

### Aggregate Loss Simulation (P&C)

The aggregate loss model combines two distributions:

**Frequency** (how many claims): `N ~ Poisson(λ)`
- λ = expected number of claims per period
- A Poisson draw might return 0, 3, 7, or 12 claims

**Severity** (how big each claim): `X ~ Lognormal(μ, σ)`
- μ = mean of the log of losses, so `e^μ` is the median loss; the mean is `e^(μ + σ²/2)`
- σ = standard deviation of the log of losses

**Aggregate Loss** for one trial:
```
L = X₁ + X₂ + ... + Xₙ  (where N is drawn from Poisson)
```

### C++ Implementation

```cpp
std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
std::poisson_distribution<int> freq_dist(expected_frequency);
std::lognormal_distribution<double> sev_dist(expected_severity_mu, severity_sigma);

for (int i = 0; i < trials; ++i) {
    int claim_count = freq_dist(gen);
    double aggregate_loss = 0.0;
    for (int c = 0; c < claim_count; ++c) {
        aggregate_loss += sev_dist(gen);
    }
    results[i] = aggregate_loss;
}
```

The generator is the Mersenne Twister, seeded from `std::random_device` when
`seed` is 0 — pass a seed for a reproducible run. Standard-library distributions
are implementation-defined, so the same seed gives different draws on different
compilers; the distribution, not the sequence, is what is portable.

### Life Portfolio Simulation

```cpp
std::normal_distribution<double> shock_dist(1.0, shock_volatility);

for (int i = 0; i < trials; ++i) {
    double shock = shock_dist(gen);                 // e.g. 1.3 = 30% mortality increase
    if (shock < 0.0) shock = 0.0;
    double actual_prob = base_mortality_rate * shock;
    if (actual_prob > 1.0) actual_prob = 1.0;

    std::binomial_distribution<int> claims_dist(policy_count, actual_prob);
    int deaths = claims_dist(gen);                  // one draw per trial, not per policy
    results[i] = deaths * death_benefit;
}
```

---

## CI/CD Pipeline

Four GitHub Actions workflows (details in [Publishing](PUBLISHING.md)):

- **CI** — every push and pull request to `main`: build, `pytest`, and the verification suite on Linux, macOS and Windows.
- **Verify** — nightly: the verification suite with the SciPy oracle, plus benchmarks.
- **Docs** — push to `main`: regenerate the chart gallery and the verification report, and publish this site.
- **Publish** — push of a `v*` tag: version guard, sdist and cibuildwheel wheels for CPython 3.10–3.14, upload to PyPI through **Trusted Publishing** (OIDC, no stored tokens), and the GitHub Release.

---

## File Structure

```
RiskPY/
├── .github/
│   ├── dependabot.yml
│   └── workflows/           ci.yml, docs.yml, publish.yml, verify.yml
├── docs/                    this site; generate_gallery.py renders the charts
├── examples/                auto_insurance_rater, batch_processing, catastrophe_model, life_annuity_pricing
├── riskpy/                  the Python package (see System Overview)
├── src/                     the C++ core (.cpp/.hpp) and bindings.cpp
├── tests/                   pytest suite
├── CMakeLists.txt           C++ build config
├── Makefile                 install, test, verify, docs, build, release
├── mkdocs.yml               docs site config
├── pyproject.toml           metadata, extras, scikit-build-core and cibuildwheel config
├── README.md                project overview (also the PyPI page)
├── RELEASE_NOTES.md
├── CONTRIBUTING.md
└── LICENSE                  MIT
```
