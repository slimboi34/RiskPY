# RiskPY: Enterprise Actuarial Engine 🚀

A hyper-fast, **C++ powered** pricing framework for actuaries and quants (P&C, Life, Health): declarative rating, Monte Carlo, chain-ladder reserving, Fourier aggregate loss, option pricing, batch Excel export, and an optional desktop GUI.

### 📖 [**Documentation & user manual → slimboi34.github.io/RiskPY**](https://slimboi34.github.io/RiskPY/)

Worked examples, every chart the library produces, and the full API — start there.

```bash
pip install open-riskpy          # core, zero dependencies
pip install open-riskpy[sim]     # + the Monte Carlo engine
pip install open-riskpy[viz]     # + the charts
```

```python
from riskpy.mc import Model, Poisson, LogNormal, Normal

model = Model(
    claim_count = Poisson(mean=140),
    severity    = LogNormal.from_moments(mean=18_000, sd=42_000),
    inflation   = Normal(mean=0.043, sd=0.012),
)

@model.formula
def annual_loss(claim_count, severity, inflation):
    return claim_count * severity * (1 + inflation)

result = model.run(200_000, seed=42)
print(result.summary())          # mean, VaR, TVaR at every level
result.plot()                     # distribution with the tail marked
```

[![CI](https://github.com/slimboi34/RiskPY/actions/workflows/ci.yml/badge.svg)](https://github.com/slimboi34/RiskPY/actions/workflows/ci.yml)
[![Docs](https://github.com/slimboi34/RiskPY/actions/workflows/docs.yml/badge.svg)](https://slimboi34.github.io/RiskPY/)
[![PyPI](https://img.shields.io/pypi/v/open-riskpy.svg)](https://pypi.org/project/open-riskpy/)
[![Python](https://img.shields.io/pypi/pyversions/open-riskpy.svg)](https://pypi.org/project/open-riskpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **v0.2.8** — a generic Monte Carlo engine (any formula, any distributions), a six-chart visualisation layer, and a quantitative finance module: Black–Scholes with Greeks and implied vol, GBM and jump-diffusion paths, portfolio VaR/ES, and Heston pricing by Fourier inversion. Plus a [documentation site](https://slimboi34.github.io/RiskPY/). See [RELEASE_NOTES.md](RELEASE_NOTES.md).

---

## Install (end users)

```bash
pip install open-riskpy
# upgrade
pip install -U open-riskpy

# The generic Monte Carlo engine (riskpy.mc) — NumPy only
pip install "open-riskpy[sim]"

# Charts (riskpy.viz) — NumPy + Matplotlib
pip install "open-riskpy[viz]"

# Optional desktop GUI
pip install "open-riskpy[gui]"
```

Pre-built wheels are published for common platforms. From source you need a **C++17** compiler and CMake (pulled automatically by the build backend when missing).

Core install has **no heavy Python deps** (matplotlib is optional via `[gui]`).  
`import riskpy` is **headless-safe** (no Tkinter required). The GUI (`UnderwritingApp`) loads only when you use it.

---

## Install (developers)

```bash
git clone https://github.com/slimboi34/RiskPY.git
cd RiskPY
make install    # pip install -e ".[dev]"
make test       # pytest
make smoke      # quick import check
# or: make check
```

Without Make:

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Core features

| Module | Role |
|--------|------|
| **FactorModel** | Multiplicative rating rules (exact + numeric bands) |
| **MonteCarloSimulator** | P&C, life, health, GBM, catastrophe simulations |
| **ActuarialMath** | PV, FV, loss ratio, mortality |
| **LossTriangle** | Chain ladder, IBNR |
| **ExperienceRating** / **ExposureRating** | Credibility, ILFs, layers |
| **RateAnalyzer** | On-level, trend, combined ratio, indicated rate change |
| **FourierTransform** | FFT / iFFT, convolution, compound-Poisson PMF |
| **UnderwritingApp** | Tkinter GUI + CSV batch → Excel |
| **RiskEngine** / **ExcelExporter** | Low-level orchestration & xlsx |

Docs: [docs/API_REFERENCE.md](docs/API_REFERENCE.md) · [docs/TUTORIALS.md](docs/TUTORIALS.md) · [docs/PUBLISHING.md](docs/PUBLISHING.md)

---

## Quick examples

### Factor pricing
```python
from riskpy import FactorModel

model = FactorModel(initial_base_rate=1000.0)
model.add_multiplier("state", "FL", 3.0)
model.add_numeric_band_multiplier("age", 16, 25, 2.0)
premium = model.calculate({"state": "FL", "age": 19.0})  # 6000.0
```

### Fourier compound Poisson (no sampling error)
```python
from riskpy import FourierTransform

# Severity always 1 → aggregate is Poisson(λ)
pmf = FourierTransform.compound_poisson_pmf(
    severity_pmf=[0.0, 1.0],
    expected_frequency=2.0,
    grid_size=64,
)
```

### Monte Carlo
```python
from riskpy import MonteCarloSimulator
import numpy as np

sim = MonteCarloSimulator(trials=100_000)
losses = sim.simulate_aggregate_loss(5.0, 10.0, 1.5)
print(np.percentile(losses, 99))
```

### Batch CSV → Excel (needs Tk/desktop for the app wrapper)
```python
from riskpy import UnderwritingApp, FactorModel

app = UnderwritingApp(title="Auto Rater", excel_template="template.xlsx")
app.add_field("state", "State", "A", choices=["NY", "CA", "FL"])
app.add_field("age", "Age", "B")
app.set_premium_column("C", "Premium")
app.set_factor_model(FactorModel(100.0))
total, n = app.calculate_batch("book.csv", "quotes.xlsx")
```

---

## Releasing to GitHub + PyPI (maintainers)

One-time: configure [PyPI Trusted Publishing](https://pypi.org/manage/project/open-riskpy/settings/publishing/) for this repo (workflow `publish.yml`, no environment name). Details in [docs/PUBLISHING.md](docs/PUBLISHING.md).

```bash
# After merging to main and bumping version in pyproject.toml:
make release   # tags vX.Y.Z and pushes — that is the whole release
```

The tag push builds the sdist and wheels, verifies them, uploads to PyPI, and
cuts the GitHub Release with artifacts attached.

| Workflow | When | What |
|----------|------|------|
| **CI** | every push/PR to `main` | build + pytest on Linux/macOS/Windows, Python 3.10–3.13 |
| **Publish** | push of a `v*` tag (or manual) | verify → sdist + cp310–cp314 wheels → PyPI → GitHub Release |

---

## Building from source (details)

| Tool | Notes |
|------|--------|
| C++17 compiler | Apple Clang, GCC ≥ 9, MSVC 2019+ |
| CMake ≥ 3.15 | Injected by the build backend if needed |
| Python ≥ 3.10 | Prebuilt wheels ship for 3.10–3.14 |
| Ninja / ccache | Optional; speeds CI and local rebuilds |

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
pip install .
pytest tests/
```

---

## License

MIT — free for personal and commercial use.

- **GitHub:** [github.com/slimboi34/RiskPY](https://github.com/slimboi34/RiskPY)
- **PyPI:** [pypi.org/project/open-riskpy](https://pypi.org/project/open-riskpy)
