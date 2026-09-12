# Credit risk

`riskpy.credit` runs from a single name to a portfolio: Merton's structural
model, hazard rates and CDS pricing, the Vasicek / Basel asymptotic single
risk factor model, a one-factor copula portfolio simulation, and rating
transition matrices.

Scalar functions are pure Python; the portfolio simulation needs NumPy.

## Expected and unexpected loss

```python
from riskpy import credit

credit.expected_loss(pd=0.02, lgd=0.45, ead=1_000_000)     # 9,000
credit.unexpected_loss(pd=0.02, lgd=0.45, ead=1_000_000)   # √(pd(1−pd))·lgd·ead
```

`unexpected_loss` treats LGD as fixed; a random LGD adds a term the docstring
spells out.

## Merton

A firm's equity is a call on its assets struck at the debt. Everything else
follows:

```python
m = credit.merton(assets=120, debt=100, T=2, r=0.03, sigma=0.25)
print(m)
```

```
distance to default 0.5086   PD 30.5512%
equity 31.2798   debt 88.7202   leverage 0.7848
credit spread 298.41 bp
```

`equity` equals `quant.black_scholes` on the assets exactly, and
`equity + debt_value == assets` — both in the verification suite. Pass `mu`
for a physical-measure distance to default; the pricing quantities stay
risk-neutral, as they should.

## Hazard rates and CDS

```python
credit.survival_probability(hazard=0.02, t=5)      # e^{-0.10}
credit.hazard_from_spread(spread=0.012, recovery=0.4)   # the credit triangle: h ≈ s/(1−R)
credit.spread_from_hazard(0.02, 0.4)

credit.cds_par_spread(hazard=0.02, recovery=0.4, r=0.03, maturity=5)   # 1.204%
credit.cds_legs(hazard, recovery, r, maturity)      # premium and protection legs
```

The par spread is the fixed rate that makes premium and protection legs
equal at inception, with a piecewise-constant hazard and the accrual
convention stated in the docstring. For small hazards it lands within 1% of
the credit triangle `h(1 − R)`, and the suite checks that it does.

## Vasicek / ASRF and Basel

The large homogeneous portfolio: one systematic factor, asset correlation
`rho`, and a closed form for the loss fraction at any confidence level.

```python
credit.asrf_conditional_pd(pd=0.02, rho=0.2, level=0.999)   # 22.63% — the stressed PD
credit.vasicek_loss_cdf(x, pd, rho)
credit.vasicek_loss_quantile(0.999, pd, rho)

credit.basel_correlation(pd, asset_class="corporate")        # 0.12–0.24, by the Basel formula
k = credit.basel_irb_capital(pd=0.01, lgd=0.45, ead=1_000_000, maturity=2.5)
print(k)
```

```
K 7.3853% of EAD   risk weight 92.32%
capital 73,853.44   RWA 923,168.01
correlation 0.1928   conditional PD 14.0273%   maturity adj 1.2598
```

That 92.32% is the number in the corporate table of the BCBS explanatory
note for PD 1%, LGD 45%, maturity 2.5 — it is pinned in the tests. Asset
classes: `"corporate"`, `"retail_mortgage"`, `"retail_revolving"`,
`"retail_other"`.

## Portfolio simulation

When the portfolio is not large, not homogeneous, or you want the whole
distribution rather than one quantile:

```python
result = credit.credit_portfolio_loss(
    pds=[0.01, 0.02, 0.05, 0.10],
    lgds=[0.4, 0.45, 0.5, 0.6],
    eads=[1e6, 2e6, 5e5, 3e5],
    rho=0.2,                 # or one value per exposure
    trials=50_000, seed=1,
)
print(result.summary())
result.var(0.999)
```

One-factor Gaussian copula: an exposure defaults when
`√ρ·Z + √(1−ρ)·ε` falls below `Φ⁻¹(pd)`. The output is a
[`riskpy.mc.Result`](monte-carlo.md), so `viz.distribution`, `viz.exceedance`
and every risk measure work on it unchanged. Its mean matches
`Σ pd·lgd·ead`, and the 99.9% VaR at `rho = 0.9` exceeds the one at
`rho = 0` — both asserted.

## Rating transitions

```python
tm = credit.sp_transition_matrix()          # a realistic one-year S&P-style matrix
tm.probability("BBB", "BB")
tm.power(5)                                  # five-year transitions
tm.cumulative_pd("BBB", 5)                   # 1.74%
tm.marginal_pd("BBB", 3)                     # default in year 3 exactly
tm.to_frame()

credit.TransitionMatrix(matrix, ratings)     # your own; rows must sum to 1, D must absorb
```

## References

- Merton, R. (1974). On the pricing of corporate debt.
- Vasicek, O. (2002). The distribution of loan portfolio value. *Risk*.
- BCBS (2005). *An Explanatory Note on the Basel II IRB Risk Weight Functions*.
- Hull, J. *Options, Futures, and Other Derivatives*, ch. 24.
