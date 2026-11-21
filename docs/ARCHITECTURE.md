# Architecture

A deep-dive into how RiskPY bridges Python and Rust to deliver actuarial computations at native speed, and where the pure-Python modelling layers sit.

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
   └── cpp_underwriter  the compiled extension (.abi3.so / .pyd)
          │
          │ PyO3  (rust/python: argument conversion, errors, GIL release)
          ▼
rust/core/        riskpy-core, plain Rust (no Python in it)
   ├── factor.rs          FactorModel: multiplicative rating
   ├── montecarlo.rs      MonteCarloSimulator: the five fixed simulations
   ├── rng.rs, dist.rs    the PCG64-DXSM stream and the samplers it feeds
   ├── actuarial.rs       ActuarialMath: annuity PV/FV, loss ratio, an illustrative mortality lookup
   ├── triangle.rs        LossTriangle: chain-ladder reserving
   ├── experience.rs      ExperienceRating: credibility and experience modification
   ├── exposure.rs        ExposureRating: increased limits factors, layer premiums, burning cost
   ├── rate.rs            RateAnalyzer: on-level premiums, trend, indicated rate change
   ├── fourier.rs         FourierTransform: FFT and the compound Poisson PMF
   ├── xlsx.rs            ExcelExporter's workbook writer (.xlsx on the zip crate)
   └── m.rs               every transcendental function, from the pure-Rust libm crate
```

`RiskEngine` and `Field` live in the bindings crate (`rust/python/src/lib.rs`),
because the engine's pricing logic is a Python callable.

The module keeps the name the C++ core had, `riskpy.cpp_underwriter`, with the same
classes, methods, argument names, defaults and exceptions, so nothing that imports it
changed when the core moved from C++ to Rust in 0.9.0.

The modelling layers (`mc` through `credit`) are ordinary Python on NumPy and do
not call into the compiled core. The compiled core has no Python dependencies at
all, which is why `import riskpy` works on a machine with nothing but the wheel
installed.

---

## The Build Pipeline

RiskPY uses [maturin](https://www.maturin.rs/) to compile the Rust core during
`pip install`. Most users never see this: pip picks a pre-built wheel whenever one
matches. The extension is built against CPython's stable ABI (`abi3`, Python 3.10
and later), so there is **one wheel per platform**: Linux x86_64 and aarch64
(manylinux), macOS arm64 and x86_64, Windows AMD64. Here's the chain for a source
build:

### 1. User runs `pip install open-riskpy`
### 2. pip reads `pyproject.toml`

```toml
[build-system]
requires = ["maturin>=1.10,<2.0"]
build-backend = "maturin"

[tool.maturin]
manifest-path = "rust/python/Cargo.toml"
module-name = "riskpy.cpp_underwriter"
```

### 3. maturin invokes cargo

cargo builds the workspace in `rust/`: `riskpy-core` (the maths) and
`riskpy-python` (the PyO3 bindings, a `cdylib` whose library name is
`cpp_underwriter`). The only crates it pulls in are PyO3, `libm`, `zip` (with its
pure-Rust deflate backend) and `getrandom`. A source build needs a Rust toolchain
(stable, via [rustup](https://rustup.rs)) and the platform linker Rust uses (Xcode
Command Line Tools on macOS, the C toolchain on Linux, the MSVC Build Tools on
Windows); no CMake and no C++ libraries to fetch.

### 4. The compiled binary is installed

The resulting `cpp_underwriter.abi3.so` (Linux, macOS) or `cpp_underwriter.pyd`
(Windows) is placed inside the `riskpy/` package directory, next to the Python
modules maturin packs from the repository's `riskpy/` folder. Python imports it as
a regular module.

---

## How the bindings work

Each class is a `#[pyclass]` in `rust/python/src/lib.rs` that wraps its
`riskpy-core` counterpart:

```rust
#[pyclass(module = "riskpy.cpp_underwriter", name = "FactorModel", subclass, weakref)]
pub struct FactorModel {
    inner: core::factor::FactorModel,
}

#[pymethods]
impl FactorModel {
    #[new]
    #[pyo3(signature = (initial_base_rate = CFloat(0.0)))]
    fn new(initial_base_rate: CFloat) -> PyResult<Self> { ... }

    #[pyo3(signature = (field_name, exact_value, multiplier))]
    fn add_multiplier(&mut self, field_name: CStr, exact_value: CStr, multiplier: CFloat) -> PyResult<()> { ... }

    #[pyo3(signature = (inputs))]
    fn calculate(&self, inputs: CInputs) -> f64 { ... }
}
```

The argument types (`CFloat`, `CInt`, `CStr`, `CFloats`, `CInputs`, in
`rust/python/src/conv.rs`) convert Python values with the rules the C++ core's
pybind11 bindings used, so every call that worked before still works and every
call that was refused is still a `TypeError`: a `float` argument takes anything with
`__float__` or `__index__`, an `int` argument refuses floats, a list argument takes
any sequence (a NumPy array included) but not `str`, and an `inputs` dict keeps a
`float` a float and a `str` a string.

When Python calls `model.calculate({"state": "FL"})`, the dict becomes a
`BTreeMap<Vec<u8>, Value>` (the `std::map<std::string, std::variant<double,
std::string>>` of the C++ core, iterated in the same key order) and is handed to
the Rust function. Errors map one to one: a refused argument is a `ValueError`
with the C++ core's message, a failed export a `RuntimeError`, an allocation that
cannot be made a `MemoryError`.

---

## How the FactorModel Engine Works

The `FactorModel` uses a simple but powerful multiplicative architecture:

```
Final Premium = Base Rate × Factor₁ × Factor₂ × Factor₃ × ...
```

### Internal Data Structures

```rust
struct ExactMatchRule {
    exact_value: Vec<u8>,      // "FL"
    multiplier: f64,           // 2.5
}

struct NumericBandRule {
    min_val: f64,              // 16
    max_val: f64,              // 25
    multiplier: f64,           // 2.0
}

// Rules keyed by field name, so a quote only touches the rules for its own fields
exact_rules: HashMap<Vec<u8>, Vec<ExactMatchRule>>,
band_rules: HashMap<Vec<u8>, Vec<NumericBandRule>>,
```

### Calculation Algorithm

```
1. Start with base_rate
2. For each (field, value) in the inputs, in key order:
   - If value is text, for each exact rule on that field:
       if value == rule.exact_value: premium *= rule.multiplier
   - If value is a number, for each band rule on that field:
       if rule.min_val <= value <= rule.max_val: premium *= rule.multiplier
3. Return premium
```

Rule evaluation is pure Rust. Each call from Python still pays the conversion of
the input `dict`, so batch work is fastest when the loop itself stays small.

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

### Rust Implementation

```rust
let mut s = Stream::new(self.seed);          // PCG64-DXSM; seed 0 draws fresh entropy
for _ in 0..self.trials {
    let claim_count = s.poisson(expected_frequency);
    let mut aggregate_loss = 0.0;
    for _ in 0..claim_count {
        aggregate_loss += s.lognormal(expected_severity_mu, severity_sigma);
    }
    results.push(aggregate_loss);
}
```

The random numbers come from a PCG64-DXSM stream with samplers written in the core
(`rust/core/src/rng.rs` and `dist.rs`), and every transcendental function goes
through the pure-Rust `libm` crate, so a seed gives the same numbers on Linux,
macOS and Windows. Pass a seed for a reproducible run; seed 0 draws a fresh one.
Seeded results changed in 0.9.0, when the core moved from the C++ standard
library's distributions (whose output differed between compilers) to this stream.
Every `simulate_*` method releases the GIL while it runs.

### Life Portfolio Simulation

```rust
for _ in 0..self.trials {
    let mut shock = s.standard_normal() * shock_volatility + 1.0;   // e.g. 1.3 = 30% mortality increase
    if shock < 0.0 { shock = 0.0; }
    let mut actual_prob = base_mortality_rate * shock;
    if actual_prob > 1.0 { actual_prob = 1.0; }

    let deaths = s.binomial(policy_count, actual_prob);              // one draw per trial, not per policy
    results.push(deaths as f64 * death_benefit);
}
```

---

## Excel export

`ExcelExporter(template=...)` keeps its `template` argument but has never read it:
the C++ core's OpenXLSX call started every export from a blank workbook, and the
Rust core does the same. Each export writes one sheet, "Sheet1", with the mapped
labels in row 1 and values from row 2 down (numbers as numbers, text as shared
strings, NaN and infinity as `#NUM!` error cells). The workbook is written beside
the target and renamed over it, so a failed export leaves an existing file
untouched.

---

## CI/CD Pipeline

Four GitHub Actions workflows (details in [Publishing](PUBLISHING.md)):

- **CI** — every push and pull request to `main`: build the wheel, install it, `pytest` and the verification suite on Linux, macOS and Windows, plus `cargo test` and clippy on the Rust core.
- **Verify** — nightly: the verification suite with the SciPy oracle, plus benchmarks.
- **Docs** — push to `main`: regenerate the chart gallery and the verification report, and publish this site.
- **Publish** — push of a `v*` tag: version guard, sdist and maturin wheels (one abi3 wheel per platform), upload to PyPI through **Trusted Publishing** (OIDC, no stored tokens), and the GitHub Release.

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
├── rust/                    the compiled core: a cargo workspace
│   ├── core/                riskpy-core, the maths in plain Rust
│   └── python/              riskpy-python, the PyO3 bindings (riskpy.cpp_underwriter)
├── tests/                   pytest suite
├── Makefile                 install, test, rust, verify, docs, build, release
├── mkdocs.yml               docs site config
├── pyproject.toml           metadata, extras, maturin config
├── README.md                project overview (also the PyPI page)
├── RELEASE_NOTES.md
├── CONTRIBUTING.md
└── LICENSE                  MIT
```
