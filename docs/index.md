# RiskPY

A C++ risk engine with a Python front door. The heavy numerics — factor models,
loss triangles, Fourier aggregate loss, exposure and experience rating — are
compiled; the modelling layer on top of them is ordinary Python you can read.

```bash
pip install open-riskpy          # core, zero dependencies
pip install open-riskpy[sim]     # + the Monte Carlo engine
pip install open-riskpy[viz]     # + the charts
```

## The idea in six lines

A Monte Carlo simulation is what you reach for when the formula is too big to
solve. Rather than integrating, you feed the formula random inputs a few hundred
thousand times and look at the distribution that comes out.

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
result.plot()
```

Your formula is called **once**, with arrays — so two hundred thousand trials is
a single vectorised expression, not a Python loop.

![Distribution with VaR and TVaR](assets/distribution.png)

## Then ask it things

```python
result.var(0.995)          # Value at Risk at 99.5%
result.tvar(0.995)         # and the mean loss beyond it
result.prob_above(5e6)     # P(annual loss > 5m)
result.sensitivity()       # which input is driving the answer
result.standard_error      # did I run enough trials?
```

## Four charts that answer four questions

<div class="grid cards" markdown>

- **Where is the tail?**

    ![](assets/exceedance.png)

    `viz.exceedance(result)` — P(loss > x), log scale, with VaR called out.

- **Did I run enough trials?**

    ![](assets/convergence.png)

    `viz.convergence(result)` — running mean with a 95% band. Still narrowing
    at the right edge means no.

- **What is driving it?**

    ![](assets/tornado.png)

    `viz.tornado(result)` — rank correlation of each input with the output.

- **How do scenarios compare?**

    ![](assets/compare.png)

    `viz.compare({...})` — same axis, one colour per scenario, always a legend.

</div>

## Quant finance

Closed-form pricing and Greeks are pure Python and need nothing installed:

```python
from riskpy import quant

quant.black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2)   # 10.4506
print(quant.greeks(S=100, K=95, T=0.5, r=0.03, sigma=0.28, kind="put"))
quant.implied_vol(price=12.5, S=100, K=100, T=1, r=0.05)
```

Path models and portfolio risk need NumPy:

```python
paths = quant.gbm_paths(S0=100, mu=0.07, sigma=0.22, T=1, trials=20_000)
viz.fan(paths)

quant.historical_var(returns, level=0.99)
quant.expected_shortfall(returns, level=0.99)
```

![GBM fan chart](assets/gbm_fan.png)

And when the model has no closed-form price but does have a characteristic
function, integrate the transform instead of simulating:

```python
quant.heston_price(S=100, K=100, T=1, r=0.03,
                   v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
```

That is the same trick the compiled `FourierTransform` class uses for aggregate
loss distributions — a transform you can write down beats a simulation you have
to wait for.

## Where to go next

- [Monte Carlo](monte-carlo.md) — the full distribution list and the `Result` API
- [Visualisation](visualisation.md) — every chart, and the rules they follow
- [Quant finance](quant.md) — pricing, Greeks, paths, portfolio risk
- [How it works](HOW_IT_WORKS.md) — the C++ core
