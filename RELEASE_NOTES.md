# RiskPY v0.3.0 — Release Notes

**Release date:** 2026-09-12  
**Type:** Feature release — four new fields of risk, a rewritten simulation engine, and a verification suite. One important fix.  
**Install:** `pip install -U open-riskpy`  
**Documentation:** <https://slimboi34.github.io/RiskPY/>

---

## TL;DR

RiskPY grows from a Monte Carlo engine with a quant module into a general risk
and actuarial library. Four new modules — **life contingencies**, **claims
reserving**, **interest rates** and **credit risk** — sit beside a rewritten
`riskpy.mc` whose twenty distributions now carry a full analytic layer, draw
Latin hypercube samples, and accept rank correlations. The chart library goes
from six charts to twenty-two. And every identity the library claims is now
checked by `riskpy.verify` — 125 checks against closed forms, published
tables and independent routes — on every push, nightly, and on the docs site.

The compiled core is untouched. `dependencies` is still empty. Nothing that
worked in 0.2.9 changed its result, with one exception below that was wrong
before and is right now.

```bash
pip install -U open-riskpy          # core, zero dependencies
pip install -U "open-riskpy[sim]"   # + NumPy
pip install -U "open-riskpy[viz]"   # + Matplotlib
```

---

## Fixed

### `quant.heston_price` returned the wrong price

The Fourier inversion evaluated the characteristic function of the log
*return* but used `log(K)` where it needed log-moneyness `log(K/S)`. At the
textbook parameters (`S = K = 100`) the price came back as roughly zero
instead of 8.93. The identity test that should have caught it — Heston
collapsing to Black–Scholes as vol-of-vol goes to zero — was in the suite and
was failing; it is now green, and the same identity is in the verification
suite so it cannot regress quietly.

### The examples did not run

`examples/catastrophe_model.py`, `examples/life_annuity_pricing.py` and three
documentation pages still called `MonteCarloSimulator.simulate_aggregate_loss(trials=…)`
as a static method, an API that changed in 0.2.4. They now construct the
simulator, and both examples run as part of checking this release.

### Smaller

- `Result.convergence()` started its running variance at one trial, where
  the sample variance is undefined, so the first band half-width was zero. It
  now starts at two and uses the unbiased estimator.
- `quant._normal_quantile` was Acklam's approximation on its own (relative
  error 1e-9). It now delegates to the new full-precision `norm_ppf`, which
  also tightened `parametric_var` and the reserving lognormal percentile.
- The light-mode chart palette failed the 3:1 contrast check on three
  colours. They are re-stepped; both palettes now pass all six categorical
  checks.

---

## Added

### `riskpy.life` — life contingencies

Pure Python, no NumPy. `LifeTable` from rates, from survivors, from Gompertz
or Makeham parameters, from an arbitrary force of mortality, or the AMLCR
Standard Ultimate Life Table. Every insurance and annuity present value —
whole life, term, pure endowment, endowment, deferred, increasing, mthly by
Woolhouse — with a `continuous=True` UDD adjustment. Net and gross premiums,
prospective reserves and reserve profiles, joint-life and last-survivor
statuses, commutation functions.

It reproduces the printed values in Dickson, Hardy & Waters to their printed
precision: `ä_40 = 18.4578`, `A_40 = 0.12106`, `l_100 = 6248.17`.

### `riskpy.reserving` — claims reserving

`Triangle` (ragged lists or NaN arrays, cumulative or incremental),
`chain_ladder` with volume, simple or regression averages and a tail,
`mack_chain_ladder` with process and parameter standard errors,
`bornhuetter_ferguson`, `cape_cod`, `expected_claims`, an England–Verrall ODP
`bootstrap_chain_ladder` whose result wraps as a `riskpy.mc.Result`, and
`fit_tail`. The GenIns triangle ships as `reserving.genins()`.

On GenIns it reproduces R's `ChainLadder` exactly: reserve 18,680,856, Mack
total standard error 2,447,095.

### `riskpy.rates` — interest rates

Discounting conventions and conversions; `YieldCurve` with linear or
log-linear interpolation, built directly, by bootstrap from par rates, or
parametrically (Nelson–Siegel, Svensson — which keep their formula, so the
limits hold exactly); `Bond` with price, yield, Macaulay and modified
duration, convexity, DV01, curve pricing and z-spread; Vasicek and CIR
closed-form bonds and curves, exact-transition Vasicek paths and
full-truncation CIR paths.

The CIR bond uses a `log1p` arrangement of the closed form that stays exact
as `σ → 0`; the textbook arrangement loses every digit below `σ ≈ 10⁻⁶`.

### `riskpy.credit` — credit risk

Expected and unexpected loss; `merton` (equity as a call on assets, distance
to default, spread); hazard rates, the credit triangle, CDS legs and par
spreads; the Vasicek / ASRF loss distribution, Basel asset correlations and
IRB capital (reproducing the BCBS table: 92.32% risk weight for PD 1%, LGD
45%); a one-factor Gaussian copula `credit_portfolio_loss` returning a
`Result`; `TransitionMatrix` with powers and cumulative default
probabilities, and a realistic S&P-style matrix to start from.

### `riskpy.mc` — rewritten

- **An analytic layer on every distribution**: `pdf`, `cdf`, `ppf`,
  `moments()` and `support()`, vectorised, scalar in → scalar out. Quantiles
  are accurate to around 1e-10 relative or better everywhere and to machine
  precision in the extreme tails — the upper tail inverts the survival
  function rather than the CDF, and heavy-shape betas and gammas whose lower
  quantiles sit near 1e-35 are seeded from the series expansion.
- **Six new distributions**: `Weibull`, `StudentT`, `Categorical`,
  `Mixture`, `Truncated`, and `Gamma.from_moments` /
  `LogNormal.from_median_cv` constructors.
- **Latin hypercube sampling**: `run(..., sampling="lhs")`. Same marginal,
  the noise on the mean cut by a factor of two to three.
- **Rank correlation**: `model.correlate("a", "b", 0.7)` or
  `simulate(..., correlation=...)`, imposed by Iman–Conover so the marginals
  are untouched. Jointly impossible correlations are rejected with the
  smallest eigenvalue named. `correlation_matrix()` and `iman_conover()` are
  public.
- **A copula hook**: `simulate(..., copula=obj)` for any object with `.dim`
  and `.uniforms(n, rng)`.
- **`Result` additions**: `median`, `skewness`, `kurtosis`, `cdf`,
  `percentiles`, `histogram`, `exceedance_curve`, `describe`,
  `sensitivity(method="pearson" | "contribution")`, `__len__`; quantile
  queries sort once and cache.
- **`Model` additions**: `correlate`, `sample`, `sweep` (re-run with one
  input swapped, same seed for every scenario).

### `riskpy.viz` — twenty-two charts

New: `density`, `cdf`, `qq`, `dashboard`, `spread`, `correlation`, `scatter`,
`triangle`, `development`, `reserve_range`, `survival`, `mortality`,
`reserve_profile`, `curve`, `allocation`, `waterfall`. Every chart is
factored so `dashboard` can draw into panels. Tick labels are compact
(`8.1m`, not `8,091,985`), `distribution` clips a heavy tail at the 99.9th
percentile and counts what it clipped, and direct labels de-collide with
leader lines. The palettes carry sequential and diverging ramps; both modes
pass the six categorical checks.

### `riskpy._special` — dependency-free special functions

Incomplete gamma and beta, the normal (with an `erfc` tail good to
`x ≈ −37`), Student-t and chi-square distributions, the Kolmogorov
distribution, digamma and trigamma, Brent, bisection, a bounded scalar
minimiser, Nelder–Mead and Simpson. Validated against SciPy to 1e-14 in the
tests without SciPy ever being a dependency. Vectorised variants iterate only
the unconverged points, which is what keeps a 200,000-point beta CDF at 37 ms.

### `riskpy.verify` — the verification suite

`verify.run()` returns a `Report`; `python -m riskpy.verify` and the
`riskpy-verify` console script print it, with `--bench` for timings and
`--json` / `--markdown` for machines. Each module contributes checks through
a `_verification_checks()` hook. It runs in CI on every push, in a new
nightly `verify.yml` workflow with the SciPy oracle and benchmarks, and the
docs workflow publishes the report.

---

## Changed

- `Poisson`, `NegativeBinomial`, `Bernoulli` and `Binomial` samples are
  float arrays (they were integer arrays), so every distribution returns the
  same dtype and formulas need no casts.
- `viz.distribution` clips its axis at the 99.9th percentile by default;
  pass `clip=None` for the full range.
- The light palette's series colours changed (contrast); the dark palette is
  as before.
- The `[dev]` extra now installs NumPy, Matplotlib and SciPy, since the test
  suite uses all three. `[docs]` requires mkdocs ≥ 1.6.
- New console script: `riskpy-verify`.
- Package summary and keywords describe the wider scope.

---

## Testing

341 tests and 125 verification checks. The new tests follow the rule the
0.2.8 notes set out — identities before reference values — and add an
external witness where one exists: the special functions, the analytic
distribution layer and the truncated normal against SciPy; reserving against
R's `ChainLadder`; life against AMLCR; credit against the BCBS table.

---

## Upgrading

No API was removed. Code written against 0.2.9 runs unchanged, with three
things worth knowing: Heston prices are now correct, integer-distribution
samples are floats, and `viz.distribution` clips its axis unless told not to.

---

# RiskPY v0.2.9 — Release Notes

**Release date:** 2026-08-31  
**Type:** Maintenance re-release — no functional changes  
**Install:** `pip install -U open-riskpy`  
**Documentation:** <https://slimboi34.github.io/RiskPY/>

v0.2.9 is functionally identical to v0.2.8. There are no source changes; the
artifacts are simply rebuilt and republished with the current toolchain. There
is no need to upgrade, and nothing changes if you do.

---

# RiskPY v0.2.8 — Release Notes

**Release date:** 2026-07-26  
**Type:** Feature release — new modules, no breaking changes  
**Install:** `pip install -U open-riskpy`  
**Documentation:** <https://slimboi34.github.io/RiskPY/>

---

## TL;DR

Three new pure-Python modules on top of the C++ core, and a documentation site.
The compiled engine is untouched, `dependencies` is still empty, and nothing
that worked in 0.2.7 changed — the new layers ship as optional extras.

```bash
pip install -U open-riskpy          # core, zero dependencies
pip install -U "open-riskpy[sim]"   # + the Monte Carlo engine
pip install -U "open-riskpy[viz]"   # + the charts
```

---

## Added

### `riskpy.mc` — generic Monte Carlo

`MonteCarloSimulator` is fast but fixed: five specific actuarial scenarios, each
with its own signature. There was no way to say *"here is my formula, here are
my uncertain inputs, go"*. That is the common case, and it is now the front door.

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
print(result.summary())
```

The formula is called **once**, with arrays — 200k trials is one vectorised
expression, not a Python loop. `vectorised=False` falls back to looping for
formulas NumPy cannot express.

Fourteen distributions. Two carry alternative constructors because the raw
parameters are never the ones you have to hand:
`LogNormal.from_moments(mean, sd)` and
`NegativeBinomial.from_mean_dispersion(mean, dispersion)`.

`Result` answers what people actually ask of a run: `.var()`, `.tvar()`,
`.percentile()`, `.prob_above()`, `.standard_error`, `.convergence()`,
`.sensitivity()`, `.summary()`, `.to_frame()`.

### `riskpy.viz` — six charts, one theme

`distribution`, `exceedance`, `convergence`, `tornado`, `fan`, `compare`. Dark
by default, light available, both palettes validated for contrast and colour
vision deficiency. Each returns a Matplotlib `Figure` and none call `show()`, so
the same call works in a notebook, a script and a test.

### `riskpy.quant` — pricing, Greeks, paths, portfolio risk

Black–Scholes, Greeks and implied volatility are **pure Python** — no NumPy — so
they work in the lean install beside the C++ core. Greeks come back in the units
people quote them in: vega per percentage point, theta per calendar day.

Implied vol uses bisection rather than Newton, because vega collapses in the
wings and Newton diverges exactly where you most want an answer.

With NumPy: `gbm_paths` and `merton_jump_paths` for path simulation,
`historical_var`, `parametric_var` and `expected_shortfall` for portfolio risk.

`heston_price` prices European options by Fourier inversion of the
characteristic function — the concrete version of *"the formula is too big to
solve, so transform it instead of simulating it"*, and the same idea the
compiled `FourierTransform` already uses for aggregate loss. It uses the "little
Heston trap" formulation, which keeps the complex logarithm on its principal
branch; the textbook form is algebraically identical and numerically wrong past
a couple of years.

### Documentation site

<https://slimboi34.github.io/RiskPY/> — worked examples, every chart the library
produces, and the full API. The gallery is regenerated from the library on every
build, so the pictures cannot drift from the code, and a broken module fails the
docs build rather than publishing a stale screenshot.

---

## Changed

- `Documentation` in the package metadata now points at the site rather than a
  README anchor. This is the link PyPI renders.
- Package summary and keywords cover the quant side: Black–Scholes, Heston,
  value at risk.
- New `[sim]`, `[viz]` and `[docs]` extras. `[gui]` is unchanged.

---

## Testing

60 new cases across `tests/test_mc.py` and `tests/test_quant.py`, on top of the
existing suite. The pricing tests lean on identities rather than only on
published values, because identities catch classes of error a single reference
number does not:

- put–call parity, exactly, for both Black–Scholes and Heston
- every Greek checked as a numerical derivative of the price
- Heston collapsing to Black–Scholes as vol-of-vol → 0
- `parametric_var` against the known normal quantile
- Merton jump-diffusion keeping its drift, which is what the compensator is for

Statistical assertions use fixed seeds and generous tolerances: a test that
fails once a month teaches people to re-run it, and then they re-run it the day
it breaks for real.

---

## Upgrading

Nothing to do. No API changed, no dependency was added to the core install. The
new modules import lazily, so `import riskpy` on a machine with no NumPy behaves
exactly as it did in 0.2.7.

---

# RiskPY v0.2.7 — Release Notes

**Release date:** 2026-07-25  
**Type:** Packaging + release-pipeline release  
**Install:** `pip install -U open-riskpy`

---

## TL;DR

Correct package metadata, wheels for **Python 3.14** and **Linux aarch64**, and a
release pipeline that verifies what it ships before it ships it. No API changes —
upgrading is safe.

```bash
pip install -U open-riskpy
python -c "import riskpy; print(riskpy.__version__)"
```

---

## Fixed

- **Broken author metadata.** `pyproject.toml` shipped a literal placeholder
  (`[EMAIL_ADDRESS]`, no `@`) as the author email. v0.2.6 on PyPI carries it;
  0.2.7 has a real address.
- **`requires-python` claimed more than we shipped.** It said `>=3.8` while
  wheels covered only cp310–cp313, so pip on 3.8/3.9 silently fell through to a
  source build needing a full C++ toolchain. Now `>=3.10`, matching the wheels.
- **Stale pybind11 floor.** The CMake `FetchContent` fallback pinned v2.11.1,
  which predates Python 3.13 support, so a build without a pip-installed
  pybind11 could not target 3.13/3.14. Floor and fallback are now 2.12 / v3.0.4.
- **`CMAKE_CXX_COMPILER_LAUNCHER=ccache` was set on all CI platforms**, including
  the Windows image, which has no ccache.

## Added

- **cp314 wheels** — Python 3.14 is supported and classified.
- **linux/aarch64 wheels**, built on a native ARM runner rather than QEMU.
- **macos/x86_64 wheels** (cross-compiled) alongside arm64.
- **Windows CI** — the platform we ship wheels for is now actually tested.
- `make wheels` — reproduce the full CI wheel matrix locally.

## Changed

- **Release is one command.** `make release` tags and pushes; the tag push
  builds, publishes to PyPI, and cuts the GitHub Release with artifacts
  attached. Previously the tag push did nothing and a GitHub Release had to be
  created by hand — despite the workflow comments claiming otherwise.
- **New pre-publish gates**, so a bad release fails before upload, not after:
  - tag-vs-`pyproject` version check (fails in seconds, not after a 15-min matrix)
  - the sdist is installed and imported from a temp dir, proving it is self-contained
  - `twine check --strict` on every artifact
  - a warning if the version is already on PyPI
- **Best-effort extra architectures.** linux/aarch64 and macos/x86_64 cannot
  block a release; core linux/macos/windows wheels still must pass.
- **cibuildwheel 2.22 → 4.1.1.** Brings `manylinux_2_28`, automatic
  `delvewheel` DLL bundling on Windows, and `abi3audit` checks. The old
  `yum || apt` compiler shim is gone — the modern image already has C++17 and
  scikit-build-core injects cmake/ninja.
- **Wheel matrix moved into `pyproject.toml`** (`[tool.cibuildwheel]`) so local
  builds and CI cannot drift apart.
- **PEP 639 licensing** — `License-Expression: MIT` (metadata 2.4) replaces the
  deprecated `license = { text = ... }` table.
- **PyPI attestations** (PEP 740) are now published with each upload.
- Explicit `MACOSX_DEPLOYMENT_TARGET=11.0` for reproducible macOS wheel tags.

---

# RiskPY v0.2.6 — Release Notes

**Release date:** 2026-07-09  
**Type:** Feature + performance + packaging release  
**Install:** `pip install -U open-riskpy`

---

## TL;DR

Faster core math, headless-safe imports, new **FourierTransform** analytic aggregate-loss engine, stronger validation, expanded tests, and a push-button **GitHub Actions → PyPI** path (CI + wheels + Trusted Publishing).

```bash
pip install -U open-riskpy
python -c "import riskpy; from riskpy import FourierTransform; print(riskpy.__version__)"
```

---

## Added

### FourierTransform (C++)
- `fft` / `ifft` — iterative radix-2 Cooley–Tukey with zero-pad to power of 2  
- `convolve` — fast linear convolution  
- `compound_poisson_pmf` — aggregate loss PMF via $\hat{S}(t)=e^{\lambda(\hat{X}(t)-1)}$  
- Severity auto-normalized; non-finite inputs rejected; empty FFT returns empty  

### Packaging / DevX
- GitHub Actions **CI** (Ubuntu + macOS, Python 3.10–3.12)  
- **Publish** workflow: sdist + cibuildwheel wheels → PyPI Trusted Publishing  
- `Makefile` (`make install|test|smoke|build|release`)  
- `docs/PUBLISHING.md` rewritten for the one-command release path  

### Tests
- Full suites for Fourier, LossTriangle, RateAnalyzer, FactorModel extras, package headless import, RiskEngine  
- Suite size: **77** tests  

---

## Changed / Fixed

### Performance
- `RateAnalyzer.on_level_premiums`: O(n²) → **O(n)**  
- `LossTriangle.get_ultimate_losses`: O(years×periods) → **O(years)** with precomputed CDFs  
- `FactorModel.calculate`: field-keyed `unordered_map` lookup  
- `RiskEngine.get_fields`: return **const reference** (zero copy in C++)  

### Stability
- `import riskpy` no longer requires Tkinter (lazy `UnderwritingApp`)  
- Factor / rate / triangle / Fourier finite-value validation  
- OpenXLSX pinned to **v0.5.1**; shallow FetchContent; ccache-aware builds  
- Excel `create(..., XLForceOverwrite)` (non-deprecated API)  

---

## Prior releases

See git history and earlier sections of this file for v0.2.5 and below.  
*(v0.2.5 on PyPI was a packaging/CMake bump; v0.2.6 is the first full Fourier + perf ship.)*

---

# RiskPY v0.2.4 — Release Notes

**Release date:** 2026-05-01  
**Type:** Patch / bug-fix release  

### Fixed
- GUI Monte Carlo dispatch (`MonteCarloSimulator` instance method)  
- Headless smoke test  
- CSV batch newline handling  

```bash
pip install --upgrade open-riskpy
```
