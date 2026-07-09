# RiskPY: Enterprise Actuarial Engine 🚀

A hyper-fast, **C++ powered** pricing framework for actuaries (P&C, Life, Health): declarative rating, Monte Carlo, chain-ladder reserving, Fourier aggregate loss, batch Excel export, and an optional desktop GUI.

```bash
pip install open-riskpy
```

```python
from riskpy import FactorModel, FourierTransform

model = FactorModel(1000.0)
model.add_multiplier("state", "FL", 3.0)
print(model.calculate({"state": "FL"}))  # 3000.0

pmf = FourierTransform.compound_poisson_pmf([0.0, 1.0], expected_frequency=2.0, grid_size=64)
```

[![CI](https://github.com/slimboi34/RiskPY/actions/workflows/ci.yml/badge.svg)](https://github.com/slimboi34/RiskPY/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/open-riskpy.svg)](https://pypi.org/project/open-riskpy/)
[![Python](https://img.shields.io/pypi/pyversions/open-riskpy.svg)](https://pypi.org/project/open-riskpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **v0.2.6** — FourierTransform, hot-path performance, headless-safe imports, wheels via GitHub Actions → PyPI. See [RELEASE_NOTES.md](RELEASE_NOTES.md).

---

## Install (end users)

```bash
pip install open-riskpy
# upgrade
pip install -U open-riskpy
```

Pre-built wheels are published for common platforms. From source you need a **C++17** compiler and CMake (pulled automatically by the build backend when missing).

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

One-time: configure [PyPI Trusted Publishing](https://pypi.org/manage/project/open-riskpy/settings/publishing/) for this repo (`publish.yml`, environment `pypi`). Details in [docs/PUBLISHING.md](docs/PUBLISHING.md).

```bash
# After merging to main and bumping version in pyproject.toml:
make release
gh release create v0.2.6 --generate-notes
# Actions → "Publish to PyPI" builds wheels + sdist and uploads
```

| Workflow | When | What |
|----------|------|------|
| **CI** | every push/PR to `main` | build + pytest (Linux/macOS) |
| **Publish** | GitHub Release / manual | wheels + sdist → PyPI |

---

## Building from source (details)

| Tool | Notes |
|------|--------|
| C++17 compiler | Apple Clang, GCC ≥ 9, MSVC 2019+ |
| CMake ≥ 3.15 | Injected by the build backend if needed |
| Python ≥ 3.8 | 3.10+ recommended |
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
