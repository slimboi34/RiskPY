# Life contingencies

`riskpy.life` is the life-and-pensions half of actuarial science: a survival
model (a life table), an interest rate, and the present-value formulas that
turn the two into insurance and annuity prices, premiums and reserves.

It is **pure Python** — `math` only — so it works in the zero-dependency
install:

```bash
pip install open-riskpy
```

```python
from riskpy import life

sult = life.LifeTable.sult()                         # the AMLCR standard table
life.whole_life_insurance(sult, 40, i=0.05)          # A_40  = 0.121059
life.whole_life_annuity_due(sult, 40, i=0.05)        # ä_40  = 18.457757
life.whole_life_premium(sult, 40, i=0.05)            # P_40  = 0.006559
```

Those three numbers are the ones printed in Dickson, Hardy & Waters,
*Actuarial Mathematics for Life Contingent Risks*, Table D.3 — which is the
point of shipping the Standard Ultimate Life Table: every worked answer in that
book is a test case.

## Conventions

- Ages and durations are **integers**. A table is a discrete survival model:
  `q_x` is the probability a life aged exactly `x` dies before `x + 1`.
- `i` is the effective annual rate as a decimal — `0.05`, not `5`.
- Insurances pay 1 at the **end of the year of death** (`A_x`); annuities-due
  pay 1 at the **start** of each year while alive (`ä_x`). Pass
  `continuous=True` for the uniform-distribution-of-deaths adjustment
  `Ā = (i/δ)·A` on insurances and Woolhouse's `ā ≈ ä − ½` on annuities. Both
  are approximations; each docstring says how good.
- A table must **close**: its last `q_x` is 1. Every identity below depends
  on it, so a table that does not close is rejected rather than truncated.

## Life tables

```python
table = life.LifeTable(qx=[0.001, 0.0012, ..., 1.0], start_age=20)
table = life.LifeTable.from_lx([100_000, 99_900, ...], start_age=20)
table = life.LifeTable.gompertz(B=2.7e-6, c=1.124)               # μ_x = B·c^x
table = life.LifeTable.makeham(A=0.00022, B=2.7e-6, c=1.124)     # μ_x = A + B·c^x
table = life.LifeTable.from_force(lambda x: 0.0002 + 2.7e-6 * 1.124 ** x)
table = life.LifeTable.sult()                                    # Makeham, ages 20–130
```

`makeham` uses the closed form for `t_p_x`, so its rates are exact;
`from_force` integrates any force of mortality with Simpson's rule (error
around 1e-12 for Gompertz-like laws at the default 64 steps).

!!! note "A steep law closes early"
    Gompertz with `c = 1.15` reaches `q_x = 1` in floating point in the
    mid-nineties. The generated table closes there, with a `RuntimeWarning`
    saying so, rather than carrying thirty ages of lives no one can reach.

```python
table.qx(40), table.px(40), table.lx(40), table.dx(40)
table.tpx(10, 40)              # 10_p_40
table.deferred_qx(10, 40)      # 10|q_40 — dies in the year after age 50
table.curtate_expectation(40)  # e_40 = 45.78 on the SULT
table.survival_curve()         # (ages, l_x) — for viz.survival
table.mortality_curve()        # (ages, q_x) — for viz.mortality
```

![Survivorship](assets/survival.png)

## Interest

```python
life.discount_factor(0.05)        # v = 0.952381
life.discount_rate(0.05)          # d = 0.047619
life.force_of_interest(0.05)      # δ = 0.048790
life.nominal_rate(0.05, m=12)     # i^(12)
life.annuity_certain(0.05, 10)    # 7.721735;  due=True → 8.107822
```

## Insurances and annuities

Every function takes `(table, x, i, …)` and returns the actuarial present
value per unit of benefit.

| Insurance | | Annuity | |
|---|---|---|---|
| `whole_life_insurance(t, x, i)` | `A_x` | `whole_life_annuity_due(t, x, i)` | `ä_x` |
| `term_insurance(t, x, n, i)` | `A¹_{x:n}` | `whole_life_annuity_immediate(t, x, i)` | `a_x` |
| `pure_endowment(t, x, n, i)` | `ₙE_x` | `temporary_annuity_due(t, x, n, i)` | `ä_{x:n}` |
| `endowment_insurance(t, x, n, i)` | `A_{x:n}` | `temporary_annuity_immediate(t, x, n, i)` | `a_{x:n}` |
| `deferred_whole_life_insurance(t, x, m, i)` | `ₘ|A_x` | `deferred_annuity_due(t, x, m, i)` | `ₘ|ä_x` |
| `increasing_insurance(t, x, i)` | `(IA)_x` | `increasing_annuity_due(t, x, i)` | `(Iä)_x` |
| | | `mthly_annuity_due(t, x, i, m)` | `ä_x^{(m)}` (Woolhouse) |

```python
life.term_insurance(sult, 40, 20, 0.05)       # 0.014633
life.pure_endowment(sult, 40, 20, 0.05)       # 0.366630
life.endowment_insurance(sult, 40, 20, 0.05)  # the sum of the two
life.mthly_annuity_due(sult, 40, 0.05, 12)    # 17.999423
```

The identities you would check by hand are checked in the verification suite:
`A_x = 1 − d·ä_x`, `A_{x:n} = A¹_{x:n} + ₙE_x`, `ä_{x:n} = ä_x − ₙE_x·ä_{x+n}`,
`a_x = ä_x − 1`, and the recursion `A_x = v·q_x + v·p_x·A_{x+1}`.

### Joint lives

Independent lives, joint-life status (fails on the first death) and
last-survivor status (fails on the second):

```python
life.joint_life_annuity_due(sult, 40, 45, 0.05)      # ä_xy  = 17.181816
life.last_survivor_annuity_due(sult, 40, 45, 0.05)   # ä_x̄ȳ = 19.092153
life.joint_life_insurance(sult, 40, 45, 0.05)
life.last_survivor_insurance(sult, 40, 45, 0.05)
```

`ä_xy + ä_x̄ȳ = ä_x + ä_y` holds exactly, and is tested.

## Commutation functions

```python
c = life.Commutation(sult, i=0.05)
c.D(60), c.N(60), c.C(60), c.M(60), c.S(60), c.R(60)
c.M(60) / c.D(60)     # == whole_life_insurance(sult, 60, 0.05)
```

The present-value functions above are computed directly; commutation functions
are here for checking against older tables and textbooks that quote them.

## Premiums

```python
life.net_premium(benefit_apv, annuity_apv)           # the equivalence principle
life.whole_life_premium(sult, 40, 0.05)              # 0.006559 per unit, payable for life
life.term_premium(sult, 40, 20, 0.05)
life.endowment_premium(sult, 40, 20, 0.05)           # 0.029343, payable for 20 years

life.gross_premium(sult, 40, 0.05, benefit=100_000,
                   initial_expense=500, renewal_expense=50,
                   premium_loading=0.03)               # 752.84 a year
```

`gross_premium` solves the equivalence principle for a whole-life contract
with an initial expense, a level renewal expense from the second year, and a
percentage-of-premium loading. The docstring states exactly which cash flows
it includes; read it before quoting the number.

## Reserves

Prospective net premium reserves, and the profile over the policy's life:

```python
life.net_premium_reserve(sult, 40, t=10, i=0.05, kind="endowment", n=20)   # 0.380073
life.reserve_profile(sult, 40, 0.05, "endowment", n=20)   # [(0, 0.0), (1, …), …, (20, 1.0)]
```

An endowment's reserve is 0 at issue and exactly 1 at maturity — both ends
are asserted in the tests, as is Fackler's recursion
`(ₜV + P)(1 + i) = q_{x+t} + p_{x+t}·ₜ₊₁V` at every duration.

![Reserve profile](assets/reserve_profile.png)

## What is not here

Select tables (mortality that depends on time since underwriting) and multiple
decrement tables are out of scope for this module. Fractional ages are not
modelled beyond the UDD and Woolhouse adjustments.

## References

- Bowers, Gerber, Hickman, Jones & Nesbitt, *Actuarial Mathematics*, 2nd ed.
- Dickson, Hardy & Waters, *Actuarial Mathematics for Life Contingent Risks*,
  2nd ed. — the source of the Standard Ultimate Life Table and the printed
  values the suite reproduces.
