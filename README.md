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
from riskpy.mc import Model, Compound, Poisson, LogNormal, Normal

model = Model(
    # 140 claims a year on average, each one sized independently
    claims    = Compound(Poisson(mean=140), LogNormal.from_moments(mean=18_000, sd=42_000)),
    inflation = Normal(mean=0.043, sd=0.012),
)
model.correlate("claims", "inflation", 0.3)

@model.formula
def annual_loss(claims, inflation):
    return claims * (1 + inflation)

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

> **v0.8.0** — fixes from a planned stress test of 0.7.0: 56 defects, every one fixed,
> plus three found while merging. Three were high severity: `Model.stress` put a 99.5%
> VaR of 20 defaults on a PD shock whose re-run gives 4, a t copula with a huge `df`
> hung `riskpy-mcp` from a 70-byte request, and the Frank sampler was wrong above
> θ ≈ 35. No single MCP request can now hang the server or exhaust its memory,
> `riskpy-mcp` honours cancellation, and text an agent supplies can no longer forge the
> `riskpy-audit` calculation record. Some results change and some inputs are now
> refused; see [RELEASE_NOTES.md](RELEASE_NOTES.md).
>
> **v0.7.0** — economic capital, stress testing and an audit trail. `riskpy.capital`
> adds five copulas with closed-form tail dependence (the `copula=` hook in `riskpy.mc`
> now has something to plug into it), a nearest-correlation repair, variance–covariance
> aggregation, and Euler allocation of TVaR or VaR to lines of business.
> `Model.stress` runs scenarios on shared random draws and `Result.reverse_stress` finds
> the scenario behind the worst outcomes. For agents: `sensitivity` and `scenario_table`
> bump and revalue any calculator, six more calculators are tools (33 in all), the MCP
> server serves resources and prompts, and `riskpy.audit` records every call to a
> hash-chained ledger that `riskpy-audit` can verify, replay and write up. No existing
> result changes. See [RELEASE_NOTES.md](RELEASE_NOTES.md).
>
> **v0.6.0** — an accuracy and edge-case pass over every module, with about forty fixes:
> exact CDS accrual, `implied_vol` 460× tighter, Student-t, chi-squared and incomplete
> gamma/beta accurate in the extreme tails, and `tvar` now expected shortfall. Also new:
> `mc.Compound` for the sum of N claims. A regression notebook runs 52 real-use scenarios
> in both versions: 0.5.2 fails 44 and 0.6.0 fails none. Some results change, and a few
> inputs now raise instead of returning a misleading number; see
> [RELEASE_NOTES.md](RELEASE_NOTES.md).
>
> **v0.5.2** — a pickled error keeps its value and notes across processes,
> `riskpy-tools trace` no longer crashes on a broken trace file, and an overflowing
> `aggregate_loss` says which argument to fix. No result changes.
>
> **v0.5.1** — fixes from black-box testing 0.5.0: parametric `value_at_risk`
> reports the normal expected shortfall (it returned the historical one),
> `bornhuetter_ferguson` refuses a loss ratio typed as a percentage, errors name
> the tool's own argument, the wheels no longer drop OpenXLSX's headers and
> libraries into `site-packages`, and macOS x86_64 wheels ship again.
> See [RELEASE_NOTES.md](RELEASE_NOTES.md).
>
> **v0.5.0** — RiskPY tells you (and your agent) exactly what went wrong.
> Every error has a stable code, names the parameter and says how to fix it
> (`riskpy-explain RP105`); the tools and the MCP server return
> `{"error": {code, message, param, hint}}` instead of tracebacks and refuse
> percentages where decimals belong (`r=5`, `sigma=20`) rather than compute a
> confident wrong answer. `riskpy-doctor` checks an installation end to end, a
> call trace shows what an agent has been doing (`riskpy-tools trace`,
> `riskpy_status`), and eleven more verified calculators are tools (27 in all).
> On a scripted set of 19 agent mistakes the share of failures that come back
> with a code, the parameter, a fix and structured JSON goes from 4% (0.4.0) to
> 100%. Ships the unpublished 0.4.1 security fixes. No numerical result changes.
> See [RELEASE_NOTES.md](RELEASE_NOTES.md).
>
> **v0.4.0** — for AI agents: sixteen verified calculators as tools with JSON
> schemas (`riskpy-tools`, `riskpy.tools.call`), an MCP server
> (`riskpy-mcp`, `pip install "open-riskpy[mcp]"`) so Claude Code and friends
> can price, reserve and simulate with numbers the verification suite stands
> behind, and `riskpy-context`, a description of the library generated from
> the installed package for the model about to use it. It builds on v0.3.x:
> four fields of risk (life, reserving, rates, credit), a rewritten Monte
> Carlo engine, twenty-two charts and the verification suite. See
> [RELEASE_NOTES.md](RELEASE_NOTES.md).

---

## What is in the box

| Module | What it does | Needs |
|---|---|---|
| **`riskpy.mc`** | Generic Monte Carlo — any formula, twenty distributions with `pdf`/`cdf`/`ppf`/`moments`, mixtures and truncation, Latin hypercube, Iman–Conover rank correlation, copulas, stress tests on shared draws, reverse stress tests, and a `Result` that answers VaR, TVaR, sensitivity and convergence | NumPy |
| **`riskpy.capital`** | Economic capital — Gaussian, Student t, Clayton, Gumbel and Frank copulas with closed-form Kendall's tau and tail dependence; nearest correlation matrix; variance–covariance aggregation; Euler, covariance and pro-rata allocation of TVaR or VaR | — / NumPy |
| **`riskpy.viz`** | Twenty-two charts, one theme, light and dark, palettes validated for contrast and colour-vision deficiency | Matplotlib |
| **`riskpy.quant`** | Black–Scholes, Greeks, implied vol (pure Python); GBM and jump-diffusion paths; historical, parametric and expected-shortfall VaR; Heston by Fourier inversion | — / NumPy |
| **`riskpy.life`** | Life tables (Gompertz, Makeham, the AMLCR standard table), insurances, annuities, premiums, reserves, joint lives, commutation functions | — |
| **`riskpy.reserving`** | Chain ladder, Mack standard errors, Bornhuetter–Ferguson, Cape Cod, ODP bootstrap, tail fitting — reproduces R's `ChainLadder` on GenIns to the unit | NumPy |
| **`riskpy.rates`** | Yield curves (bootstrap, Nelson–Siegel, Svensson), bonds with duration, convexity, DV01, z-spread; Vasicek and CIR with closed-form bonds and simulated paths | — / NumPy |
| **`riskpy.credit`** | Merton, hazard rates and CDS, Vasicek / Basel ASRF with IRB capital, one-factor copula portfolio simulation, rating transition matrices | — / NumPy |
| **`riskpy.verify`** | The verification suite: 125 checks against closed forms, published tables and identities — 139 with SciPy installed as an oracle — plus benchmarks. `riskpy-verify` on the command line | NumPy |
| **`riskpy.tools`**, **`riskpy.audit`** | 33 verified calculators as agent tools (CLI, Python, MCP) with coded errors; a hash-chained ledger of every call that can be verified, replayed and reported | — |
| **Compiled core** | `FactorModel`, `MonteCarloSimulator`, `LossTriangle`, `FourierTransform`, `ExperienceRating`, `ExposureRating`, `RateAnalyzer`, `ActuarialMath`, `RiskEngine`, `ExcelExporter`, the optional Tkinter `UnderwritingApp` | — |

Every layer is imported lazily, so `import riskpy` on a machine with nothing
but the wheel installed works and gives you the compiled core.

---

## A taste of each

```python
from riskpy import life, reserving, rates, credit, quant, capital, viz

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

# Capital: dependence, aggregation, allocation
capital.StudentTCopula(0.5, df=4).tail_dependence().upper            # 0.2532; a Gaussian copula has 0
agg = capital.aggregate_capital({"market": 100, "credit": 60, "insurance": 80},
                                {("market", "credit"): 0.5, ("market", "insurance"): 0.25,
                                 ("credit", "insurance"): 0.25})
agg.total, agg.diversification_benefit                               # 180.0, 60.0
book = Model(motor=LogNormal.from_moments(100, 20), property=LogNormal.from_moments(60, 30),
             liability=LogNormal.from_moments(40, 25))
losses = book.sample(400_000, seed=1, copula=capital.GumbelCopula(1.5, dim=3))
capital.allocate(losses, level=0.995).contributions                  # Euler TVaR by line; adds up to the total

# Stress: scenarios on the same draws, and the scenario behind the worst outcomes (the model from the top)
stress = model.stress({"inflation 8%": {"inflation": 0.08}}, trials=100_000, seed=42)
stress.delta("inflation 8%", "var", 0.995)
model.run(100_000, seed=42).reverse_stress(0.995).scenario

# Quant: still pure Python for the closed forms
quant.black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2)           # 10.4506
quant.heston_price(S=100, K=100, T=1, r=0.03, v0=0.04, kappa=2, theta=0.04, xi=0.5, rho=-0.7)
```

---

## For AI agents

```bash
riskpy-tools                         # 33 verified calculators as tools, with JSON schemas
riskpy-tools call black_scholes '{"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}'   # 10.4506
riskpy-mcp                           # the same tools over MCP  (pip install "open-riskpy[mcp]")
riskpy-context                       # the library described for the model about to use it
riskpy-doctor                        # is this installation healthy? (--json; exit 1 on a failure)
riskpy-explain RP104                 # what an error code means and how to fix the call
riskpy-tools trace                   # the recent tool calls, with durations and error codes
riskpy-mcp --audit run.jsonl         # record every call, in full, to a hash-chained ledger
riskpy-audit replay run.jsonl        # recompute every recorded call and report any drift
```

A call that goes wrong comes back as data an agent can act on, never a traceback:

```bash
$ riskpy-tools call black_scholes '{"S": 100, "K": 100, "T": 1, "r": 5, "sigma": 0.2}'
{"error": {"code": "RP104", "param": "r",
           "message": "r = 5 looks like a percentage: rates are decimals, so |r| must be at most 1",
           "hint": "Divide by 100 — 5% is 0.05.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp104"}}
```

`from riskpy import tools; tools.manifest()` is the list as data and `tools.call(name, {...})`
runs one; every tool wraps a public function the verification suite checks, takes plain
numbers (rates as decimals, times in years) and returns JSON. `riskpy-mcp --config` prints
the client configuration for Claude Code, Claude Desktop or Cursor. `riskpy-context`
generates the conventions that bite, every module's public signatures and the tool list
from the installed package, so it cannot go stale. Over MCP, `riskpy_status`, `riskpy_doctor`
and `riskpy_explain` let the agent see its recent calls, check the installation and look up a
code or a tool's formula.

Two tools drive the others. `sensitivity` bumps one input of any calculator and returns the
derivative, second derivative and elasticity of every output; `scenario_table` runs any
calculator under named what-ifs:

```bash
$ riskpy-tools call sensitivity '{"tool": "black_scholes", "parameter": "S",
    "arguments": {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}}'
{"tool": "black_scholes", "parameter": "S", "value": 100.0, "bump": 0.01, "scheme": "central",
 "outputs": {"result": {"base": 10.450583572185565, "derivative": 0.6368306425770243,
                        "second_derivative": 0.018762017077733617, "elasticity": 6.093732834900834, ...}}}
```

With `RISKPY_AUDIT_FILE` set (or `riskpy-mcp --audit FILE`), every call is appended to a
ledger with its full arguments and result, each entry chained to the last by SHA-256.
`riskpy_audit` gives the agent the head digest to quote with its numbers, `riskpy-audit
verify` shows whether anything was altered, `riskpy-audit replay` recomputes every call, and
`riskpy-audit report` writes the calculation record. The MCP server also serves resources
(`riskpy://conventions`, `riskpy://errors`, …) and prompts (`capital_model`,
`reserve_review`, `stress_test`, `audit_trail`). See [the docs](https://slimboi34.github.io/RiskPY/agents/),
[diagnostics](https://slimboi34.github.io/RiskPY/diagnostics/), the
[audit ledger](https://slimboi34.github.io/RiskPY/audit/) and the
[error codes](https://slimboi34.github.io/RiskPY/errors/).

## Verification

```bash
riskpy-verify              # 125 checks (139 with SciPy), a second or two
riskpy-verify --bench      # plus timings
riskpy-verify -m life      # one module; a module whose extra is missing is skipped, not failed
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
make test       # pytest — 522 test cases
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
| **Publish** | push of a `v*` tag | version guard → sdist + cp310–cp314 wheels (import-tested) → PyPI → GitHub Release |

---

## License

RiskPY is Copyright © 2026 Joshua Harty and is released under the **MIT License** — see
[LICENSE](LICENSE). Free for personal and commercial use, modification and redistribution;
keep the copyright and permission notice with any copy.

- **GitHub:** [github.com/slimboi34/RiskPY](https://github.com/slimboi34/RiskPY)
- **PyPI:** [pypi.org/project/open-riskpy](https://pypi.org/project/open-riskpy)
