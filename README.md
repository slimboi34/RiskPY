# RiskPY

A risk and actuarial engine: a **C++ core** with no Python dependencies, and a
modelling layer in readable Python that covers Monte Carlo with dependence,
life contingencies, claims reserving, yield curves and short-rate models,
credit risk, option pricing, and twenty-two charts — with a **verification
suite** that checks every identity the library claims on every push.

### 📖 [**Documentation → slimboi34.github.io/RiskPY**](https://slimboi34.github.io/RiskPY/)

```bash
pip install open-riskpy            # core, zero dependencies
pip install "open-riskpy[sim]"     # + NumPy: simulation and the numeric layers
pip install "open-riskpy[viz]"     # + Matplotlib: the charts
```

```python
from riskpy.mc import Model, Poisson, LogNormal, Normal

model = Model(
    claim_count = Poisson(mean=140),
    severity    = LogNormal.from_moments(mean=18_000, sd=42_000),
    inflation   = Normal(mean=0.043, sd=0.012),
)
model.correlate("claim_count", "inflation", 0.3)

@model.formula
def annual_loss(claim_count, severity, inflation):
    return claim_count * severity * (1 + inflation)

result = model.run(200_000, seed=42, sampling="lhs")
print(result.summary())          # mean, VaR, TVaR at every level
result.plot("dashboard")         # distribution, exceedance, convergence, tornado
```

[![CI](https://github.com/slimboi34/RiskPY/actions/workflows/ci.yml/badge.svg)](https://github.com/slimboi34/RiskPY/actions/workflows/ci.yml)
[![Verify](https://github.com/slimboi34/RiskPY/actions/workflows/verify.yml/badge.svg)](https://github.com/slimboi34/RiskPY/actions/workflows/verify.yml)
[![Docs](https://github.com/slimboi34/RiskPY/actions/workflows/docs.yml/badge.svg)](https://slimboi34.github.io/RiskPY/)
[![PyPI](https://img.shields.io/pypi/v/open-riskpy.svg)](https://pypi.org/project/open-riskpy/)
[![Python](https://img.shields.io/pypi/pyversions/open-riskpy.svg)](https://pypi.org/project/open-riskpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **v0.3.0** — four new fields of risk (life, reserving, rates, credit), a
> rewritten Monte Carlo engine with an analytic layer on every distribution,
> Latin hypercube sampling and rank correlation, twenty-two charts, a
> dependency-free special-function layer, and the verification suite. Also
> fixes a Heston mispricing that had been there since 0.2.8. See
> [RELEASE_NOTES.md](RELEASE_NOTES.md).

---

## What is in the box

| Module | What it does | Needs |
|---|---|---|
| **`riskpy.mc`** | Generic Monte Carlo — any formula, twenty distributions with `pdf`/`cdf`/`ppf`/`moments`, mixtures and truncation, Latin hypercube, Iman–Conover rank correlation, a copula hook, and a `Result` that answers VaR, TVaR, sensitivity and convergence | NumPy |
| **`riskpy.viz`** | Twenty-two charts, one theme, light and dark, palettes validated for contrast and colour-vision deficiency | Matplotlib |
| **`riskpy.quant`** | Black–Scholes, Greeks, implied vol (pure Python); GBM and jump-diffusion paths; historical, parametric and expected-shortfall VaR; Heston by Fourier inversion | — / NumPy |
| **`riskpy.life`** | Life tables (Gompertz, Makeham, the AMLCR standard table), insurances, annuities, premiums, reserves, joint lives, commutation functions | — |
| **`riskpy.reserving`** | Chain ladder, Mack standard errors, Bornhuetter–Ferguson, Cape Cod, ODP bootstrap, tail fitting — reproduces R's `ChainLadder` on GenIns to the unit | NumPy |
| **`riskpy.rates`** | Yield curves (bootstrap, Nelson–Siegel, Svensson), bonds with duration, convexity, DV01, z-spread; Vasicek and CIR with closed-form bonds and simulated paths | — / NumPy |
| **`riskpy.credit`** | Merton, hazard rates and CDS, Vasicek / Basel ASRF with IRB capital, one-factor copula portfolio simulation, rating transition matrices | — / NumPy |
| **`riskpy.verify`** | The verification suite: 111 checks against closed forms, published tables and identities — 125 with SciPy installed as an oracle — plus benchmarks. `riskpy-verify` on the command line | NumPy |
| **Compiled core** | `FactorModel`, `MonteCarloSimulator`, `LossTriangle`, `FourierTransform`, `ExperienceRating`, `ExposureRating`, `RateAnalyzer`, `ActuarialMath`, `RiskEngine`, `ExcelExporter`, the optional Tkinter `UnderwritingApp` | — |

Every layer is imported lazily, so `import riskpy` on a machine with nothing
but the wheel installed works and gives you the compiled core.

---

## A taste of each

```python
from riskpy import life, reserving, rates, credit, quant, viz

# Life: the textbook's own numbers
sult = life.LifeTable.sult()
life.whole_life_annuity_due(sult, 40, i=0.05)        # ä_40 = 18.457757 (AMLCR Table D.3)
life.net_premium_reserve(sult, 40, t=10, i=0.05, kind="endowment", n=20)

# Reserving: the GenIns triangle, with uncertainty
tri = reserving.genins()
mack = reserving.mack_chain_ladder(tri)
mack.total_reserve, mack.total_se                     # 18,680,856 ± 2,447,095
boot = reserving.bootstrap_chain_ladder(tri, n=5000, seed=1)
viz.exceedance(boot.result())                         # a reserve distribution, charted

# Rates: a bond on a bootstrapped curve
curve = rates.YieldCurve.bootstrap([0.020, 0.025, 0.028, 0.030], [1, 2, 3, 4])
bond = rates.Bond(face=100, coupon=0.05, maturity=4, frequency=2)
bond.z_spread(bond.price(0.045), curve)

# Credit: Basel capital and a correlated portfolio
credit.basel_irb_capital(pd=0.01, lgd=0.45, ead=1e6).risk_weight    # 92.32%
pds, lgds, eads = [0.01] * 500, [0.45] * 500, [1e6] * 500             # 500 identical loans
credit.credit_portfolio_loss(pds, lgds, eads, rho=0.2, trials=100_000, seed=1).var(0.999)

# Quant: still pure Python for the closed forms
quant.black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2)           # 10.4506
quant.heston_price(S=100, K=100, T=1, r=0.03, v0=0.04, kappa=2, theta=0.04, xi=0.5, rho=-0.7)
```

---

## Verification

```bash
riskpy-verify              # 111 checks (125 with SciPy), ~1 second
riskpy-verify --bench      # plus timings
```

A test suite says the code does what it did yesterday. The verification suite
says it is *right*: put–call parity, `A_x = 1 − d·ä_x`, Mack's standard error
on the reference triangle, the Basel risk-weight table, `Φ⁻¹(Φ(x)) = x`, the
contrast ratio of every palette colour — each computed and compared to a
source the library had no hand in. It runs in CI on every push, nightly with
the SciPy oracle, and the docs site publishes the latest report.

---

## Install (developers)

```bash
git clone https://github.com/slimboi34/RiskPY.git
cd RiskPY
make install    # pip install -e ".[dev]"
make test       # pytest — 407 test cases
make verify     # the verification suite
```

Without Make:

```bash
pip install -e ".[dev]"
pytest tests/
python -m riskpy.verify
```

From source you need a **C++17** compiler and CMake ≥ 3.15 (pulled
automatically by the build backend when missing). Pre-built wheels ship for
Python 3.10–3.14 on Linux, macOS and Windows.

---

## The compiled core

```python
from riskpy import FactorModel, FourierTransform, MonteCarloSimulator, LossTriangle

model = FactorModel(initial_base_rate=1000.0)
model.add_multiplier("state", "FL", 3.0)
model.add_numeric_band_multiplier("age", 16, 25, 2.0)
model.calculate({"state": "FL", "age": 19.0})          # 6000.0

# Severity always 1 → aggregate is Poisson(λ), with no sampling error at all
pmf = FourierTransform.compound_poisson_pmf([0.0, 1.0], expected_frequency=2.0, grid_size=64)

sim = MonteCarloSimulator(trials=200_000, seed=7)
losses = sim.simulate_aggregate_loss(5.0, 8.0, 0.5)
```

`import riskpy` is headless-safe; the Tkinter GUI (`UnderwritingApp`) loads
only when you use it.

---

## Releasing (maintainers)

```bash
# After merging to main and bumping version in pyproject.toml:
make release   # tags vX.Y.Z and pushes — that is the whole release
```

The tag push checks the tag against the version, builds the sdist and wheels
and smoke-tests their imports, uploads to PyPI via Trusted Publishing, and cuts
the GitHub Release with artifacts attached.
Details in [docs/PUBLISHING.md](docs/PUBLISHING.md).

| Workflow | When | What |
|----------|------|------|
| **CI** | every push/PR to `main` | build + pytest + verification on Linux/macOS/Windows, Python 3.10–3.13 |
| **Verify** | nightly | verification with the SciPy oracle, benchmarks, JSON report as an artifact |
| **Docs** | push to `main` | regenerate the gallery and the verification report, build and publish the site |
| **Publish** | push of a `v*` tag | verify → sdist + cp310–cp314 wheels → PyPI → GitHub Release |

---

## License

MIT — free for personal and commercial use.

- **GitHub:** [github.com/slimboi34/RiskPY](https://github.com/slimboi34/RiskPY)
- **PyPI:** [pypi.org/project/open-riskpy](https://pypi.org/project/open-riskpy)
