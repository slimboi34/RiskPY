# Economic capital

`riskpy.capital` covers the three steps of a capital model: how the risks move
together (copulas), how much capital the whole needs (aggregation), and whose
capital it is (allocation). *New in 0.7.0.*

```python
from riskpy.mc import Model, LogNormal
from riskpy.capital import GumbelCopula, allocate

book = Model(
    motor=LogNormal.from_moments(100, 20),
    property=LogNormal.from_moments(60, 30),
    liability=LogNormal.from_moments(40, 25),
)
losses = book.sample(400_000, seed=1, copula=GumbelCopula(1.5, dim=3))
print(allocate(losses, level=0.995).summary())
```

```text
TVaR 0.995 allocation (euler, 400,000 trials)
Line           Mean  Standalone TVaR  Contribution   Share
---------  --------  ---------------  ------------  ------
motor       99.9990         173.8765      166.3741   31.2%
property    59.9968         212.8107      200.7690   37.6%
liability   39.9041         178.9358      166.6437   31.2%
Total      199.8999         565.6230      533.7867  100.0%
Diversification benefit 31.8362 (5.6% of the standalone sum)
```

The closed forms (Kendall's tau, tail dependence, copula CDFs,
`aggregate_capital`) are pure Python. Sampling, `allocate` and
`nearest_correlation` need NumPy (`pip install "open-riskpy[sim]"`).

**Conventions.** Losses are positive numbers: bigger is worse. Confidence
levels are decimals (`0.995`). TVaR means expected shortfall, the same
definition as `Result.tvar`. A copula's `correlation` is the correlation of its
latent normal or t variables, which is not the Spearman rank correlation that
`Model.correlate` takes; `kendall_tau()` and `spearman_rho()` convert.

## Copulas

A correlation says how often two risks are bad together on an ordinary day. It
does not say whether they are bad together in the tail, and capital is set in
the tail. A copula describes the whole dependence, separately from each risk's
own distribution.

| Copula | Parameter | Kendall's tau | Tail dependence |
|---|---|---|---|
| `GaussianCopula(correlation)` | ρ in [−1, 1], or a matrix | 2/π · asin ρ | none |
| `StudentTCopula(correlation, df)` | ρ or a matrix; `df` > 0 | 2/π · asin ρ | both tails: 2·t<sub>df+1</sub>(−√((df+1)(1−ρ)/(1+ρ))) |
| `ClaytonCopula(theta, dim=2)` | θ > 0 | θ/(θ+2) | lower: 2<sup>−1/θ</sup> |
| `GumbelCopula(theta, dim=2)` | θ ≥ 1 | 1 − 1/θ | upper: 2 − 2<sup>1/θ</sup> |
| `FrankCopula(theta, dim=2)` | θ ≠ 0 | 1 − 4/θ · (1 − D₁(θ)) | none |

`ClaytonCopula` and `GumbelCopula` take `survival=True`, which swaps the tails:
a survival Clayton has its dependence among the large losses.

Every copula has:

- `dim` and `uniforms(n, rng)`, which is what `Model.run(copula=...)` uses;
- `sample(n, seed=None)`, an `(n, dim)` array of dependent uniforms;
- `kendall_tau()` and `tail_dependence()`, in closed form;
- `cdf(u, v)`, the probability that both are below their `u` and `v` quantiles;
- `joint_exceedance(level)`, the probability that both exceed the same quantile;
- `from_kendall_tau(tau, ...)`, to build one from a rank correlation.

For the Gaussian and t copulas with more than two risks, `kendall_tau`,
`tail_dependence`, `cdf` and `joint_exceedance` take `i` and `j` to choose the
pair. `GaussianCopula` also has `spearman_rho()`.

### The same rank correlation, five different tails

All five copulas below have Kendall's tau = 0.5. The last two columns are the
probability that both risks exceed their own 1-in-200 level in the same year,
and how many times more likely that is than under independence (2.5 in
100,000).

```python
from riskpy.capital import GaussianCopula, StudentTCopula, GumbelCopula, ClaytonCopula, FrankCopula

for cop in (FrankCopula.from_kendall_tau(0.5), GaussianCopula.from_kendall_tau(0.5),
            StudentTCopula.from_kendall_tau(0.5, df=4), GumbelCopula.from_kendall_tau(0.5),
            ClaytonCopula.from_kendall_tau(0.5, survival=True)):
    print(cop, cop.tail_dependence().upper, cop.joint_exceedance(0.995))
```

| Copula | Upper tail dependence | P(both beyond 99.5%) | × independence |
|---|---|---|---|
| Frank (θ = 5.74) | 0 | 0.000140 | 5.6 |
| Gaussian (ρ = 0.707) | 0 | 0.001171 | 46.9 |
| Student t (ρ = 0.707, df = 4) | 0.397 | 0.002108 | 84.3 |
| Gumbel (θ = 2) | 0.586 | 0.002936 | 117.5 |
| Survival Clayton (θ = 2) | 0.707 | 0.003536 | 141.4 |

Choosing the copula is a modelling decision with a price. For the three-line
book at the top of this page, at the same seed and trial count, the 99.5% TVaR
is 446.7 under a Gaussian copula with correlations 0.5 / 0.25 / 0.25, 482.9
under a t copula with the same matrix and 4 degrees of freedom, and 533.8 under
the Gumbel copula shown.

### In a model

```python
model.run(100_000, seed=1, copula=StudentTCopula(0.5, df=4))
model.sample(100_000, seed=1, copula=GaussianCopula([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]]))
```

The model's variables take the copula's columns in the order they were
declared, each through its own quantile function, so the marginals are exactly
the distributions you gave. A variable with no quantile function (a `Compound`)
is simulated and its values are placed in the order of the copula's column: the ranks are the copula's and the marginal is the simulated
one. Pass either `copula=` or correlations (`model.correlate`), not both.

To write your own copula, subclass `capital.Copula` and implement
`uniforms(n, rng)`.

### Checking simulated data

```python
from riskpy.capital import empirical_tail_dependence

u = GumbelCopula(2.0).sample(400_000, seed=1)
empirical_tail_dependence(u[:, 0], u[:, 1], level=0.99)     # TailDependence(lower≈0.15, upper≈0.58)
```

This is the conditional probability at a finite quantile, P(V > q | U > q). It
tends to the copula's coefficient only as the level approaches 1, so compare it
with `copula.joint_exceedance(level) / (1 − level)`.

## A correlation matrix that cannot exist

A matrix assembled from expert judgement or pairwise estimates is often not
positive semi-definite: the pairwise numbers cannot all hold at once, and
nothing can be simulated from them. `nearest_correlation` returns the closest
matrix that can (Higham 2002, alternating projections with Dykstra's
correction). A valid matrix comes back unchanged.

```python
from riskpy.capital import nearest_correlation

nearest_correlation([[1, 1, 0], [1, 1, 1], [0, 1, 1]])      # Higham's own example
# [[1.     0.7607 0.1573]
#  [0.7607 1.     0.7607]
#  [0.1573 0.7607 1.    ]]
```

## Aggregation from standalone figures

When all you have is each risk's standalone capital and a correlation matrix,
`aggregate_capital` applies the variance–covariance formula, total = √(c′Rc).
This is the formula the Solvency II standard formula uses between modules.

```python
from riskpy.capital import aggregate_capital

agg = aggregate_capital(
    {"market": 100.0, "credit": 60.0, "insurance": 80.0},
    {("market", "credit"): 0.5, ("market", "insurance"): 0.25, ("credit", "insurance"): 0.25},
)
print(agg.summary())
```

```text
Risk       Standalone  Contribution   Share
---------  ----------  ------------  ------
market       100.0000       83.3333   46.3%
credit        60.0000       43.3333   24.1%
insurance     80.0000       53.3333   29.6%
Total        240.0000      180.0000  100.0%
Diversification benefit 60.0000 (25.0% of the standalone sum)
```

`standalone` is a `{name: amount}` mapping or a list (with optional `names=`).
`correlation` is a matrix in the same order or `{(a, b): rho}` pairs; unlisted
pairs are independent. The result has `total`, `undiversified`,
`diversification_benefit`, `diversification_ratio`, `contributions`, `shares`,
`summary()`, `to_dict()` and `plot()`.

The contributions are the Euler allocation, cᵢ·(Rc)ᵢ / total, and add up to the
total exactly. The formula is exact when the risks are jointly elliptical and
every standalone figure is the same multiple of a standard deviation. For
skewed risks or tail dependence it is an approximation; simulate and use
`allocate`. No NumPy needed.

## Allocation from simulated losses

`allocate(losses, level=0.99, measure="tvar", method="euler")` takes the
simulated losses of each line and returns an `Allocation`.

`losses` is `{name: array}`, a `(trials, lines)` array with `names=`, or a
DataFrame. The arrays must come from one joint simulation, so that entry *k* of
every array is the same trial. `Model.sample(...)` and `Result.inputs` are
already in that form.

| `method` | What each line gets |
|---|---|
| `"euler"` (default) | Its marginal contribution. For TVaR this is the co-TVaR, E[Xᵢ ∣ S in the worst 1 − level]. For VaR it is E[Xᵢ ∣ S = VaR], estimated from the trials whose total is nearest the VaR. |
| `"covariance"` | E[Xᵢ] + Cov(Xᵢ, S)/Var(S) · (total − E[S]). The same as Euler for jointly normal losses. |
| `"standalone"` | Pro rata to its own risk measure. |

The result has `total` (the risk measure of the summed losses, equal to
`Result.tvar(level)` on the same sample), `standalone`, `contributions`,
`means`, `shares`, `excess` (contribution above the line's own mean),
`diversification_benefit`, `diversification_ratio`, `summary()`, `to_dict()`,
`to_frame()` and `plot()` (the [allocation chart](visualisation.md), which
needs Matplotlib).

Under the Euler principle a line's return on its allocated capital moves the
same way as the portfolio's when the line grows, so it is the allocation that
gives consistent pricing signals. The co-TVaR identity holds exactly on the
sample: the contributions add up to the total to rounding error. A point mass
at the quantile is counted only for the part of it that lies in the tail, as in
`Result.tvar`.

The methods disagree when the lines have different tails. For the Gumbel book
at the top (99.5% TVaR, total 533.79):

| Method | motor | property | liability |
|---|---|---|---|
| euler | 166.37 | 200.77 | 166.64 |
| covariance | 183.73 | 199.29 | 150.76 |
| standalone | 164.09 | 200.83 | 168.86 |

Euler VaR contributions are much noisier than co-TVaR, because they average
over a thin slice of trials rather than the whole tail. `bandwidth=` sets the
share of trials in that slice.

## As agent tools

| Tool | What it does | Needs |
|---|---|---|
| `capital_aggregate` | `aggregate_capital` on a list of standalone figures and a correlation matrix | — |
| `capital_allocation` | simulates lognormal or normal lines joined by a copula, then `allocate` | NumPy |
| `copula_dependence` | Kendall's tau, tail dependence and the joint exceedance of a two-risk copula | — |
| `nearest_correlation` | repairs a correlation matrix; reports whether it changed and by how much | NumPy |

```bash
riskpy-tools call capital_aggregate '{"standalone": [100, 60, 80], "names": ["market", "credit", "insurance"],
  "correlation": [[1, 0.5, 0.25], [0.5, 1, 0.25], [0.25, 0.25, 1]]}'
riskpy-tools call copula_dependence '{"family": "gumbel", "copula_theta": 2}'
riskpy-tools call capital_allocation '{"means": [100, 60, 40], "sds": [20, 30, 25], "copula": "gumbel",
  "copula_theta": 1.5, "level": 0.995, "trials": 400000, "seed": 1}'
```

`capital_allocation` takes each line's mean and standard deviation on the money
scale. Gaussian and t copulas take `correlation` (and `df`); Clayton, Gumbel
and Frank take `copula_theta`. Pass a non-zero `seed` for a reproducible
answer. A matrix that is not positive semi-definite is refused with RP100 and a
hint naming `nearest_correlation`.

## How it is verified

`riskpy-verify -m capital` runs 14 checks. `tests/test_capital.py` checks, among
others:

- the Gaussian copula CDF against SciPy's bivariate normal and Sheppard's
  formula, and the t copula CDF against a chi-square mixture integral;
- every sampler: uniform marginals (Kolmogorov–Smirnov), Kendall's tau against
  its closed form, and joint probabilities against the copula CDF;
- `nearest_correlation` against Higham's published example;
- `aggregate_capital` against arithmetic done by hand;
- `allocate` on a ten-trial sample worked by hand, and against the closed form
  for jointly normal losses, μᵢ + Cov(Xᵢ, S)/σ<sub>S</sub> · φ(z)/(1 − α), on a
  million trials.
