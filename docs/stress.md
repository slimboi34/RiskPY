# Stress and scenario testing

Four tools for the question "what if?". *New in 0.7.0.*

| | Asks | Where |
|---|---|---|
| `Model.stress(...)` | What is the loss distribution under each scenario? | `riskpy.mc` |
| `Result.reverse_stress(...)` | Which scenario produces the extreme outcomes? | `riskpy.mc` |
| `sensitivity` | How fast does a calculator's output change with one input? | agent tool |
| `scenario_table` | What does a calculator return under each named what-if? | agent tool |

## Stress testing a model

`Model.stress` re-evaluates the formula with shocked inputs on the *same random
draws* as the base run.

```python
from riskpy.mc import Model, Compound, LogNormal, Normal, Poisson

model = Model(
    claims=Compound(Poisson(mean=120), LogNormal(mu=8.5, sigma=1.2)),
    inflation=Normal(mean=0.04, sd=0.01),
)
model.correlate("claims", "inflation", 0.3)

@model.formula
def annual_loss(claims, inflation):
    return claims * (1 + inflation)

stress = model.stress({
    "inflation 8%": {"inflation": 0.08},
    "claims +20%":  {"claims": lambda x: 1.2 * x},
    "heavier tail": {"claims": Compound(Poisson(120), LogNormal(8.5, 1.4))},
}, trials=100_000, seed=42)

print(stress.summary(level=0.995))
```

```text
Stress test — annual_loss, 100,000 trials, level 0.995

  scenario                  mean             VaR            TVaR          Δ mean           Δ VaR
  base              1,260,374.29    2,062,312.36    2,244,163.57
  inflation 8%      1,308,130.02    2,121,149.59    2,308,773.17      +47,755.73      +58,837.23
  claims +20%       1,512,449.14    2,474,774.83    2,692,996.29     +252,074.86     +412,462.47
  heavier tail      1,636,108.99    3,161,626.83    3,745,405.77     +375,734.70   +1,099,314.48
```

`scenarios` maps each label to `{variable: shock}`. A shock is one of three
things:

| Shock | Effect |
|---|---|
| a number | The variable is fixed at that value in every trial. Python and NumPy numbers both work (`0.08`, `np.float32(0.08)`, `np.int64(2)`); `True` and `False` are refused. |
| a callable | The draws are transformed: it takes the array and returns an array of the same length. It gets a copy, so a function that changes the array in place (`x *= 2`) cannot leak into the base or another scenario. |
| a `Distribution` | The variable is redrawn from it at the same ranks as the base draw. The trial that had the worst claims still has the worst claims, so every correlation and copula is preserved. |

The remaining arguments (`trials`, `seed`, `fn`, `vectorised`, `sampling`,
`copula`) are those of `Model.run`, and the base is the `Model.run` of the same
seed, trial for trial.

The formula cannot change the draws either. It is given them read-only, and a
formula that changes an input in place (`claims -= deductible`, where
`claims = claims - deductible` was meant) is run again on copies. It gets the
answer it would get on fresh arrays, and the base, every scenario and
`Result.inputs` keep the draws as drawn. *Changed in 0.9.0:* the formula was
given the draws themselves, so such a formula applied the deductible twice in
every scenario and left its scratch values in the base's `inputs`. `Model.run`
and `simulate` protect the draws the same way.

### How a distribution shock keeps the dependence

Under `copula=`, the new values are the shock's quantiles at the copula's own
uniforms. That is exactly the column a re-run with the new marginal, the same
copula and the same seed would draw, so a stressed default flag, claim count or
severity carries the copula's dependence value for value.

Otherwise (independent inputs, or `correlate`), a fresh sample of the shock is
sorted into the base ranks. Inputs with repeated values need care here: a
default flag is 0 in most trials, a claim count repeats, a low-frequency line
has many claim-free years. Trials that tie in the base are ordered by what drew
them: the normal score that `correlate` paired them by, or an independent
tie-break for an independent input. They are never ordered by trial number.
Before 0.8.0 they were, which dropped the dependence inside every tie group and
put all the extra defaults of a portfolio-wide PD shock on the same trials. With
20 independent obligors stressed from a 2% to a 5% PD, the 99.5% VaR came out at
20 defaults instead of 4.

The fresh sample comes from a random stream of its own for each variable,
derived from the seed. A scenario's numbers therefore do not depend on which
other scenarios are listed with it, or in what order, and two scenarios that
redraw a variable from the same distribution use the same values.

The result is a `StressResult`:

- `base` and `scenarios[label]` are ordinary `Result` objects;
- `delta(label, statistic="mean", level=0.995)` is scenario minus base, for
  `"mean"`, `"std"`, `"var"` or `"tvar"`;
- `worst(statistic="tvar", level=0.995)` names the scenario with the highest
  value;
- `to_dict(level)`, `to_frame(level)` and `summary(level)` tabulate everything.

### Why shared draws matter

Running the model twice, once with each assumption, puts sampling noise from
both runs into the difference. With shared draws the noise cancels. Measured on
the model above with inflation shifted by one point, over 20 seeds of 100,000
trials: the estimated change in the mean (about 12,100) had a standard
deviation across seeds of 1,215 with two independent runs and 8.5 with
`stress`. The script is `benchmarks/capital/run.py`.

## Reverse stress testing

A stress test starts from a scenario and asks for the outcome. A reverse stress
test starts from the outcome. `Result.reverse_stress` takes the trials where
the output breached a level and reports what the inputs looked like in them.

```python
result = model.run(100_000, seed=42)
rs = result.reverse_stress(0.995)
print(rs.summary())
```

```text
Reverse stress — annual_loss >= 2.06231e+06 (the worst 0.500%)
  trials in tail  500 of 100,000   mean output there 2.24416e+06

  input                       mean   in the tail  shift (sd)  percentile
  claims               1.21123e+06   2.13658e+06       +4.07       99.8%
  inflation              0.0399921      0.050392       +1.04       85.1%
```

- `reverse_stress(level=0.995)` looks at the worst `1 − level` of trials:
  500 of 100,000. The tail is weighted the way `Result.tvar` weights it.
  Trials beyond the 99.5% quantile count in full. Trials tied *at* the quantile
  (a loss that is often exactly zero, or capped at a limit) share whatever
  weight is left. So the mean output in the tail is `result.tvar(level)`.
  `threshold=` takes an output value instead, and counts every trial at or
  beyond it in full. `tail="lower"` looks at the low end (profits, solvency
  ratios).
- `rs.scenario` is `{input: mean in the tail}`, the average scenario behind the
  extreme outcomes.
- `rs.drivers` lists the inputs by how far they moved from their own mean, in
  standard deviations.
- `rs.inputs[name]` holds `mean`, `tail_mean`, `shift_sd` and `percentile` (the
  share of that input's own draws at or below its tail mean).
- `rs.probability`, `rs.tail_trials`, `rs.threshold` and `rs.output_mean`
  describe the tail; `to_dict()` gives it all as data.

The run must have kept its inputs (`keep_inputs=True`, the default).

The tail mean of an input is the conditional expectation E[input ∣ output
beyond the threshold]. For a sum of normal inputs that has a closed form,
μᵢ + Cov(Xᵢ, S)/σ<sub>S</sub> · φ(z)/(1 − α), which is what the tests compare
it with.

## Sensitivities of any calculator

The `sensitivity` tool bumps one input of any calculator and revalues. It
returns the first and second derivative and the elasticity of every numeric
output.

```bash
riskpy-tools call sensitivity '{"tool": "black_scholes", "parameter": "S",
  "arguments": {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}}'
```

```json
{"tool": "black_scholes", "parameter": "S", "value": 100.0, "bump": 0.01, "scheme": "central",
 "outputs": {"result": {"base": 10.450583572185565, "down": 10.444216203860648, "up": 10.456952816712189,
                        "derivative": 0.6368306425770243, "second_derivative": 0.018762017077733617,
                        "elasticity": 6.093732834900834}}}
```

The derivative is the option's delta (0.63683) and the second derivative its
gamma (0.018762).

| Argument | Meaning |
|---|---|
| `tool` | the calculator to differentiate |
| `arguments` | its base-case arguments, as an object |
| `parameter` | the numeric argument to bump |
| `bump` | the bump in the parameter's own units; default 0.0001 × max(1, ∣value∣) for a closed form, 1% of ∣value∣ for a simulation (1% of the distance to 0 or 1 for a confidence level), or 1 for a whole-number parameter |
| `relative` | `true` to read `bump` as a fraction of the value (0.01 is a 1% bump) |

Things to know:

- **Units.** Derivatives are per one whole unit of the input. Multiply by
  0.0001 for one basis point of a rate and by 0.01 for one point of volatility.
- **Outputs** are keyed by dotted path: `price`, `var.0.995`, or `result` when
  the calculator returns a bare number.
- **Elasticity** is derivative × value / base, the percentage change in the
  output per 1% change in the input. It is `null` when the base output is zero.
- **Edges.** If one bumped value is refused (volatility below zero, say), the
  tool uses a three-point one-sided difference and reports
  `"scheme": "forward"` or `"backward"`.
- **Simulations** need a non-zero `seed` in `arguments`, so that every
  revaluation uses the same draws. Without one the call is refused (RP113).
  Even with the seed fixed, a simulated output moves in steps, so the default
  bump for a simulation is 1% of the value rather than 0.0001 of it; repeat
  with another bump before relying on a simulated derivative.
- **Budget.** The base run and the bumped runs together may ask for no more
  work than one call of the calculator at its limit (RP405), and the same
  holds for the base and the scenarios of a `scenario_table`.

## Scenario tables for any calculator

`scenario_table` runs a calculator under named sets of overrides and reports
each result against the base case.

```python
from riskpy import tools

table = tools.call("scenario_table", {
    "tool": "basel_irb_capital",
    "arguments": {"pd": 0.01, "lgd": 0.45, "ead": 1_000_000},
    "scenarios": {"downturn": {"pd": 0.03, "lgd": 0.6}, "typo": {"pd": 3}},
})
table["base"]["result"]["capital"]                               # 73,853.44
table["scenarios"]["downturn"]["change"]["capital"]              # +63,146.82
table["scenarios"]["downturn"]["relative_change"]["capital"]     # 0.855
table["scenarios"]["typo"]["error"]["code"]                      # 'RP103'
table["failed"]                                                  # 1
```

Each scenario's overrides replace the base arguments. For every numeric output
the table gives `change` (scenario minus base) and `relative_change` (as a
decimal). A scenario the calculator refuses is reported under its own label
with the coded error, and the others still run. An override that the calculator
does not take fails the whole call with RP402 and a suggestion. One call takes
at most 100 scenarios.

Both tools work on every calculator in `riskpy-tools`, need nothing beyond what
the calculator itself needs, and are available over MCP.
