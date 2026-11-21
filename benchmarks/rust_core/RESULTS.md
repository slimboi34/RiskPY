# The Rust core against the C++ core (0.9.0 vs 0.8.0)

Measured on 2026-11-21 on an Apple M4 (10 cores), macOS 26, CPython 3.12.13. "C++" is
`open-riskpy==0.8.0` from PyPI (the macOS arm64 wheel); "Rust" is this tree built with
`maturin build --release` (fat LTO). Each side ran in its own interpreter.

```bash
cargo build --release --manifest-path rust/Cargo.toml -p riskpy-core --example sample
python benchmarks/rust_core/samplers.py    rust/target/release/examples/sample
python benchmarks/rust_core/differential.py --cpp VENV_CPP/bin/python --rust VENV_RUST/bin/python --examples 1000
python benchmarks/rust_core/simulations.py  --cpp VENV_CPP/bin/python --rust VENV_RUST/bin/python
python benchmarks/rust_core/throughput.py   --cpp VENV_CPP/bin/python --rust VENV_RUST/bin/python --rounds 7
```

## 1. Differential test of every deterministic method (`differential.py`)

Hypothesis generated 1,000 examples per property (wide floats with 0, −0, subnormals,
±1e300, ±inf and NaN mixed in; ints across and beyond the C++ `int` range; bools, `None`,
strings, bytes, NumPy scalars and arrays, `Decimal`s; empty, ragged and non-finite lists),
and every call went to both cores: **76,116 calls, no unexplained mismatch**.

| class | methods compared | calls | unexplained mismatches | bit-identical floats | worst relative difference |
|---|---|---:|---:|---:|---:|
| ActuarialMath | calculate_loss_ratio, future_value, lookup_mortality_rate, present_value | 6,000 | 0 | 1,468 / 1,468 | identical apart from the intended changes |
| ExcelExporter | constructor | 1,000 | 0 | — | — |
| ExperienceRating | constructor, buhlmann_straub_credibility, calculate_credibility, credibility_weighted_rate, expected_losses, experience_mod_factor | 8,000 | 0 | 1,448 / 1,448 | identical apart from 11 FMA cases (below) |
| ExposureRating | constructor, burning_cost, excess_risk_load, increased_limits_factor, layer_premium, loss_elimination_ratio | 8,000 | 0 | 1,259 / 1,314 | 2.7e-16 |
| FactorModel | constructor, add_multiplier, add_numeric_band_multiplier, calculate, set_base_rate | 7,787 | 0 | 901 / 901 | identical |
| Field | constructor | 2,000 | 0 | — | — |
| FourierTransform | compound_poisson_pmf, convolve, fft, ifft | 4,000 | 0 | 16,801,810 / 16,816,369 | 3.4e-16 of the input's size |
| LossTriangle | add_origin_year, get_development_factors, get_ultimate_losses, get_ibnr_reserves, get_origin_year_count, get_development_period_count | 9,558 | 0 | 2,664 / 2,664 | identical |
| MonteCarloSimulator | argument checks of all five `simulate_*` and their deterministic corners | 13,771 | 0 | 492,469 / 498,738 | 4.9e-15 (12-step paths at zero volatility) |
| RateAnalyzer | constructor and all eight methods | 12,000 | 0 | 4,887 / 4,893 | 1.9e-16 |
| RiskEngine | execute, get_fields, export_to_excel, export_batch_to_excel (cells read back) | 4,000 | 0 | — | identical cells |

Exceptions were compared too: the same type in the same cases, and the same message for
every `ValueError` and for the `RuntimeError` of an invalid Excel column. `TypeError`
messages differ (pybind11's "incompatible function arguments … Invoked with" text is
gone), the type does not.

Where the C++ was pure IEEE arithmetic the results are bit-identical. Where it called the
C library (pow, erfc, sin/cos in the FFT, exp in a zero-volatility path) the Rust core
calls the pure-Rust `libm` instead, so the last bit can differ; the tolerance was 1e-12
relative to the size of the terms and the worst case was 4.9e-15.

**Intended differences**, each checked rather than waved through:

| what | cases | how it was judged |
|---|---:|---|
| `present_value` / `future_value` with negative periods now raise `ValueError` | 114 | the 0.8.0 known limit |
| … with a NaN or infinite payment now raise `ValueError` | 6 | the 0.8.0 known limit |
| … differ from 0.8.0 by more than 1e-12 | 1,308 | Rust within a few ulps (× the condition number of exp) of a 50-digit mpmath value, and the C++ error larger |
| … differ in the last bits | 263 | Rust no further from the 50-digit value than the C++ |
| `experience_mod_factor` one ulp apart | 11 | the C++ value equals `fma(ballast, expected, actual)/…` and the Rust value the unfused expression (below) |

How much the annuity forms gained, against the same 50-digit values (4,000 random calls
of `present_value` per row, n from 1 to 1,200, payments from 1 to 1e7):

| rates | 0.8.0 worst relative error | calls off by more than 1e-12 | 0.9.0 worst relative error |
|---|---:|---:|---:|
| ordinary, −0.5 to 1 | 2.0e-12 | 1 | 1.0e-13 |
| tiny, ±1e-16 to ±1e-5 | 1.0 (every digit lost) | 3,985 | 3.7e-16 |

**The C++ core was not platform-independent even in plain arithmetic.** The macOS
arm64 wheel was compiled by clang, which contracts `a*b + c` into one fused multiply-add
by default. In 20,000 random calls each, `experience_mod_factor` differed between the
fused and the unfused evaluation 2,283 times and `credibility_weighted_rate` 3,443
times, and the 0.8.0 arm64 wheel matched the fused value every time. The x86_64 wheels
(Linux, Windows; baseline x86-64 has no FMA instruction) compute the unfused value,
which is what Rust always computes. Not checked on an x86_64 machine here (none available).

## 2. The samplers against their closed forms (`samplers.py`)

400,000 draws each, straight from the core; a fresh seed per row.

| distribution | test | p-value | z(mean) | z(var) |
|---|---|---:|---:|---:|
| uniform [0, 1) | KS | 0.49 | −1.03 | −1.57 |
| normal (polar method) | KS | 0.73 | −0.30 | −0.34 |
| lognormal (0, 1), (8, 0.5), (−3, 2) | KS on the log | 0.67, 0.029, 0.63 | ≤ 1.81 | ≤ 0.82 |
| gamma, shape 0.01, 0.1, 0.5, 1, 2.5, 30 | KS | 0.54, 0.27, 0.71, 0.68, 0.48, 0.46 | ≤ 0.96 | ≤ 1.19 |
| gamma, shape 1e4, 1e8, 1e12 | KS (standardised) | 0.0056, 0.29, 0.64 | ≤ 2.54 | ≤ 1.50 |
| Poisson λ = 0.5, 3, 9.99 (inversion) | χ² | 0.81, 0.77, 0.29 | ≤ 1.33 | ≤ 1.56 |
| Poisson λ = 10, 25, 1e3, 1e6, 1e9 (PTRS) | χ² | 0.029, 0.15, 0.012, 0.13, 0.91 | ≤ 2.69 | ≤ 2.27 |
| binomial (10, 0.3), (100, 0.2), (50, 0.9) (inversion) | χ² | 0.32, 0.90, 0.38 | ≤ 1.80 | ≤ 0.90 |
| binomial (1000, 0.3), (1000, 0.95), (1e6, 0.5), (2e9, 1e-6), (2³¹−1, 0.5) (BTPE) | χ² | 0.57, 0.26, 0.85, 0.83, 0.60 | ≤ 2.07 | ≤ 2.19 |
| binomial (5, 1), (7, 0) | exact | — | — | — |

All 32 pass (p > 1e-4, |z| < 5). The two lowest p-values were re-run with 8 seeds × 2M
draws: gamma(1e4) and Poisson(10) gave Fisher-combined p = 0.98 and 0.77.

PCG64-DXSM itself: 1,000 outputs for each of the seeds 1, 12345 and 4294967295 equal
NumPy's `PCG64DXSM` set to the same internal state, to the bit.

## 3. The simulations against closed forms and against 0.8.0 (`simulations.py`)

200,000 trials per side.

| method | parameters | z(mean) | z(var) | exact test (p) | KS Rust vs C++ (p) | q99 / q99.5 Rust÷C++ |
|---|---|---:|---:|---|---:|---|
| aggregate | λ 3, severity ≡ 1 | 0.24 | −0.13 | χ² vs Poisson counts (0.92) | 0.75 | 1.000 / 1.000 |
| aggregate | λ 5, LN(8, 0.5) | 1.01 | 0.73 | | 0.71 | 0.999 / 0.999 |
| aggregate | λ 140, LN(9.3, 1.1) | −0.25 | −0.20 | | 0.79 | 0.997 / 0.999 |
| aggregate | λ 0.2, LN(2, 2) | 0.02 | 0.07 | | 1.00 | 0.986 / 1.008 |
| life | n 1000, q 0.01, no shock | 1.08 | −1.75 | χ² vs binomial (0.51) | 1.00 | 1.000 / 1.000 |
| life | n 50,000, q 0.004, shock 0.25 | 0.77 | 2.21 | | 0.88 | 1.003 / 1.000 |
| life | n 200, q 0.6, shock 0.5 | 0.06 | 0.20 | | 0.98 | 1.000 / 1.000 |
| economic | 1 step, μ 0.05, σ 0.2 | −0.75 | −0.62 | KS of log vs normal (0.30) | 0.56 | 1.002 / 0.999 |
| economic | 12 steps, μ −0.02, σ 0.35 | −0.80 | −0.90 | KS of log vs normal (0.70) | 0.47 | 0.989 / 0.991 |
| health | λ 10, d 0.3, Γ(2, 500) | −0.06 | −1.11 | | 0.45 | 0.991 / 0.989 |
| health | λ 4, d 2.5, Γ(0.7, 1000) | 0.52 | 0.42 | | 0.89 | 1.003 / 1.003 |
| health | λ 10, d 1e-300 (Poisson limit) | 0.01 | −1.09 | | 0.58 | 1.001 / 1.000 |
| catastrophe | p 0.05, Pareto(1e6, 1.8) | (∞ var) | | KS sizes vs Pareto (0.47) | 1.00 | 0.970 / 0.990 |
| catastrophe | p 0.6, Pareto(10, 3.5) | 0.12 | 0.05 | KS sizes vs Pareto (0.44) | 0.77 | 0.998 / 1.002 |

The moments of the shocked life portfolio come from integrating the binomial over the
floored normal shock; the health moments from the gamma–Poisson mixture
(Var N = λ + dλ²). A seed repeats exactly and seed 0 does not, for every method.

Extreme parameters behave as in 0.8.0: λ = 1e9 with a large dispersion raises the same
`ValueError`; a dispersion of 1e300, a severity shape of 1e-300 or a base mortality of 0
with a 1e308 shock give all zeros on both; α = 1e-300 gives ∞ on both; n = 2³¹ − 1
policies runs on both.

## 4. Throughput (`throughput.py`)

Median of 7 rounds; each round is the best of 3 calls in a fresh interpreter, the two
cores alternating. Times include building the returned Python list.

| workload | trials | C++ ms | Rust ms | C++ ÷ Rust |
|---|---:|---:|---:|---:|
| aggregate loss λ=5, lognormal | 200,000 | 31.6 | 25.5 | 1.24× |
| aggregate loss λ=140 (PTRS) | 50,000 | 128.3 | 111.7 | 1.15× |
| life portfolio n=1000, shock 0.1 (inversion) | 200,000 | 22.0 | 26.9 | 0.82× |
| life portfolio n=50,000 (BTPE) | 200,000 | 28.0 | 16.5 | 1.70× |
| economic path, 12 steps | 200,000 | 98.3 | 38.4 | 2.56× |
| health claims λ=10, d=0.3 | 200,000 | 144.5 | 63.1 | 2.29× |
| catastrophe p=0.05 | 1,000,000 | 21.4 | 17.2 | 1.24× |
| compound_poisson_pmf, λ=50, 201-point severity | — | 0.70 | 0.80 | 0.87× |

The two slower rows spend their time in `exp`/`log1p` (a binomial set up per trial)
and `sin`/`cos` (the FFT twiddles), which the Rust core takes from the portable `libm`
rather than Apple's faster platform library: the price of the same numbers everywhere.
Only this machine was measured.
