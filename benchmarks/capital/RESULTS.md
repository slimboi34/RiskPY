# Capital, stress and audit benchmark — RiskPY 0.8.0

Produced by `benchmarks/capital/run.py` (its docstring says what each section measures). NumPy 2.5.5, SciPy 1.18.1, Python 3.12.13.

## Copula samplers against their closed forms

1,000,000 draws each, seed 1, every family at Kendall's tau = 0.5. The Kendall's tau column uses a 200,000-draw subsample (SciPy's estimator). `z` is the gap between observed and exact joint exceedance in binomial standard errors; `z, 10 seeds` pools seeds 1 to 10 (ten million draws).

| copula | tau (exact) | tau (sample) | P(both > 0.99) exact | observed | z | z, 10 seeds | ms per million draws |
|---|---|---|---|---|---|---|---|
| Gaussian | 0.5000 | 0.5004 | 0.002735 | 0.002648 | -1.66 | -2.00 | 102 |
| Student t, df = 4 | 0.5000 | 0.4999 | 0.004323 | 0.004495 | +2.62 | +0.68 | 155 |
| Clayton (survival) | 0.5000 | 0.5005 | 0.007071 | 0.006951 | -1.44 | -0.41 | 46 |
| Gumbel | 0.5000 | 0.4994 | 0.005887 | 0.005927 | +0.52 | -1.53 | 69 |
| Frank | 0.5000 | 0.5000 | 0.000544 | 0.000571 | +1.14 | +0.05 | 22 |

## Euler TVaR contributions against the normal closed form

X ~ N(10, 2²), Y ~ N(5, 1²), correlation 0.5, level 0.99. Exact contributions: x = 15.0368, y = 7.0147 (total 22.0515). Worst absolute error of either contribution over 10 seeds, and the largest gap between the sum of the contributions and the total.

| trials | worst error | as % of the smaller contribution | max ∣Σ contributions − total∣ | ms per allocation |
|---|---|---|---|---|
| 10,000 | 0.1323 | 1.89% | 7.1e-15 | 0.8 |
| 100,000 | 0.0680 | 0.97% | 1.1e-14 | 8.7 |
| 1,000,000 | 0.0209 | 0.30% | 2.8e-14 | 96.9 |
| 4,000,000 | 0.0083 | 0.12% | 5.0e-14 | 422.9 |

## Shared draws in a stress test

Compound Poisson(120) × LogNormal(8.5, 1.2) claims, inflation N(4%, 1%), rank correlation 0.3; loss = claims × (1 + inflation). The effect on the mean loss of inflation one point higher, 100,000 trials, 20 seeds.

| how the difference is estimated | mean of the estimate | standard deviation across seeds |
|---|---|---|
| two independent runs | 12,206 | 1,214.8 |
| `Model.stress` (shared draws) | 12,115 | 8.5 |

Noise ratio: 144×.

## Timings

Median of repeated runs on the machine that produced this file.

| operation | time |
|---|---|
| `tools.call("black_scholes")`, no ledger | 19.9 µs |
| the same, recorded to an in-memory ledger | 28.6 µs |
| the same, recorded to a ledger file | 58.0 µs |
| `audit.verify_chain` on 10,000 entries | 65 ms |
| `audit.replay` on 10,000 entries | 351 ms |
| `sensitivity` of `black_scholes` (three valuations) | 114 µs |
| `scenario_table` of `black_scholes`, 20 scenarios | 293 µs |
| `capital_allocation`, 3 lognormal lines, Gumbel copula, 200,000 trials | 236 ms |
