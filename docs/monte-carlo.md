# Monte Carlo

`riskpy.mc` is for the case where the formula is too awkward to solve. Instead
of integrating, you feed it random inputs a few hundred thousand times and read
the distribution that falls out.

```bash
pip install "open-riskpy[sim]"
```

## The shape of it

```python
from riskpy.mc import Model, Poisson, LogNormal, Normal, PERT

model = Model(
    claim_count          = Poisson(mean=140),
    severity             = LogNormal.from_moments(mean=18_000, sd=42_000),
    inflation            = Normal(mean=0.043, sd=0.012),
    reinsurance_recovery = PERT(low=0.0, mode=0.15, high=0.45),
)

@model.formula
def annual_loss(claim_count, severity, inflation, reinsurance_recovery):
    gross = claim_count * severity * (1 + inflation)
    return gross * (1 - reinsurance_recovery)

result = model.run(200_000, seed=42)
```

Your formula is called **once**, with NumPy arrays of length `trials` — not once
per trial. Two hundred thousand trials is a single vectorised expression.

If your formula has branching NumPy cannot express, pass `vectorised=False` and
it is looped instead. Correct, just slower.

```python
result = model.run(50_000, vectorised=False)
```

!!! tip "Always pass a seed"
    `seed=42` makes a run reproducible. Without one you get a different answer
    every time, which is fine for exploring and miserable for a report someone
    else has to check.

## Without the decorator

`simulate()` is the same engine with no `Model` object, which reads better when
you are sweeping scenarios in a loop:

```python
from riskpy.mc import simulate, Constant

result = simulate(
    lambda claims, severity: claims * severity,
    trials=100_000,
    seed=42,
    label="annual loss",
    claims=Poisson(mean=140),
    severity=LogNormal.from_moments(mean=18_000, sd=42_000),
)
```

## Distributions

| Continuous | Discrete | Other |
|---|---|---|
| `Normal(mean, sd)` | `Poisson(mean)` | `Constant(value)` |
| `LogNormal(mu, sigma)` | `NegativeBinomial(n_success, p)` | `Empirical(data)` |
| `Uniform(low, high)` | `Bernoulli(p)` | |
| `Triangular(low, mode, high)` | `Binomial(trials, p)` | |
| `PERT(low, mode, high, lam=4)` | | |
| `Exponential(scale)` | | |
| `Gamma(shape, scale)` | | |
| `Beta(a, b)` | | |
| `Pareto(xm, alpha)` | | |

Two of these have alternative constructors, and they exist because the raw
parameters are never the numbers you actually have:

```python
# You know the mean and standard deviation of severity — not mu and sigma
# of the underlying normal. Getting this backwards is the most common
# severity-modelling mistake there is.
LogNormal.from_moments(mean=18_000, sd=42_000)

# You know claims average 140 a year and are more variable than Poisson.
NegativeBinomial.from_mean_dispersion(mean=140, dispersion=2.4)
```

!!! warning "Pareto with a light tail index"
    `Pareto` has an infinite mean for `alpha <= 1` and infinite variance for
    `alpha <= 2`. The sample mean will still print a number. It will not mean
    anything.

Parameters are validated when the distribution is constructed, not when it is
sampled — a negative standard deviation is a typo, and finding out after a
ten-second run is worse than useless.

## Reading the result

```python
result.mean                  # 2,684,912.44
result.std
result.standard_error        # how much the mean would move on a new seed
result.percentile(99.5)
result.var(0.995)            # Value at Risk
result.tvar(0.995)           # mean loss beyond VaR
result.prob_above(5e6)       # P(annual loss > 5m)
result.prob_below(1e6)
result.values                # the raw array, if you want to do your own thing
```

`summary()` prints the lot:

```python
print(result.summary())
```

```
Monte Carlo — annual_loss
  trials       200,000   seed 42
  mean         2,684,912.44   ± 8,441.02 (95% CI)
  std dev      1,926,334.10
  min / max    142,880.55 / 41,204,776.31

  level        VaR              TVaR
  50.000%        2,281,443.02      3,614,882.19
  99.500%       10,884,201.77     13,996,455.30
```

### VaR and TVaR

`var(0.995)` is the loss exceeded 0.5% of the time. `tvar(0.995)` is the
*average* of those worst 0.5% of outcomes, so it is always at least as large.

Prefer TVaR for the reason regulators do: VaR tells you where the threshold is
and nothing whatsoever about how bad things get past it. Two portfolios with
identical VaR can have wildly different tails.

## Did I run enough trials?

`standard_error` is how much the mean would wobble on a different seed. It
shrinks with the square root of the trial count, so ten times the precision
costs a hundred times the work.

```python
n, running_mean, half_width = result.convergence()
result.plot("convergence")
```

If the band is still visibly narrowing at the right-hand edge of that chart, the
answer is no.

## What is driving the answer?

```python
for name, correlation in result.sensitivity():
    print(f"{name:24} {correlation:+.2f}")
```

```
severity                 +0.71
claim_count              +0.44
reinsurance_recovery     -0.38
inflation                +0.04
```

This is Spearman **rank** correlation, not Pearson. The relationship between an
input and a total is usually monotonic and rarely linear, and rank correlation
does not care about the shape. Sign gives direction: `reinsurance_recovery`
going up pushes the loss down, as it should.

Plot it with `result.plot("tornado")`.

!!! note
    Sensitivity needs the sampled inputs, which are kept by default. If you ran
    with `keep_inputs=False` to save memory, this raises rather than guessing.

## Exporting

```python
result.to_frame()      # pandas DataFrame: every input column plus the output
```

## Common errors, and what they mean

**"The formula returned shape (), expected (200000,)"** — your formula collapsed
the arrays to a scalar. Usually an `if` (use `np.where`), a `min()` (use
`np.minimum`), or a `sum()` that should not be there.

**"12,431 of 200,000 trials produced NaN or infinity"** — nearly always a
division by a variable that can reach zero, or a log of one that can go
negative. The count tells you how much of the tail is involved.

**"claim_count must be a Distribution, got float"** — wrap fixed numbers in
`Constant(...)`. This is deliberate: silently accepting a bare number makes it
easy to freeze an input you meant to vary.
