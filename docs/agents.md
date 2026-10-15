# For AI agents

An agent that can call functions needs three things from a library: a list of what it may
call, a schema for each call, and results it can pass on as data. RiskPY provides all three,
generated from the calculators' own signatures and docstrings, plus a description of the
library written for the model that is about to use it. Since 0.5.0 it also tells the agent
exactly what went wrong when a call fails — a stable error code, the parameter, the fix, as
JSON — and lets it look at what has been happening ([Diagnostics](diagnostics.md)).

```bash
pip install open-riskpy              # the core: most tools run with no dependencies
pip install "open-riskpy[sim]"       # + NumPy: the claims-triangle and VaR tools
pip install "open-riskpy[mcp]"       # + the MCP SDK: riskpy-mcp
```

## The tools

```bash
riskpy-tools                         # the list
riskpy-tools --json                  # the manifest: name, description, input schema, what it needs
riskpy-tools call black_scholes '{"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}'   # 10.4506
riskpy-tools schema aggregate_loss
riskpy-tools trace                   # recent calls, durations, error codes
riskpy-doctor                        # is this installation healthy?
riskpy-explain RP104                 # what an error code means
```

```python
from riskpy import tools

tools.manifest()                                     # a list of dicts, one per tool
tools.call("basel_irb_capital", {"pd": 0.01, "lgd": 0.45, "ead": 1e6})["risk_weight"]   # 0.9232
tools.call("black_scholes", {"S": 100, "K": 100, "T": 1, "r": 5, "sigma": 0.2}, errors="return")
# {"error": {"code": "RP104", "param": "r", "message": "r = 5 looks like a percentage: ...",
#            "hint": "Divide by 100 — 5% is 0.05.", "docs": "..."}}
```

| tool | what it does |
|---|---|
| `black_scholes`, `option_greeks`, `implied_vol` | European options, pure Python (`black_scholes` takes `explain: true` for d1, d2 and the convention) |
| `value_at_risk` | historical or parametric VaR and expected shortfall of a return series (NumPy) — *new in 0.5.0* |
| `bond_analytics` | price, Macaulay and modified duration, convexity, DV01 at a yield |
| `convert_rate`, `bootstrap_yield_curve`, `short_rate_zero_bond` | compounding conventions, a zero curve from par yields, Vasicek / CIR bond prices — *new in 0.5.0* |
| `basel_irb_capital`, `expected_loss`, `merton`, `cds_par_spread` | credit risk |
| `hazard_from_spread`, `vasicek_loss_quantile` | the credit triangle; the ASRF portfolio loss quantile — *new in 0.5.0* |
| `annuity_certain`, `life_annuity_due`, `life_insurance`, `net_premium_reserve` | life contingencies on the AMLCR standard table |
| `term_insurance`, `endowment_insurance`, `temporary_annuity_due`, `life_premium` | n-year contracts and net premiums — *new in 0.5.0* |
| `aggregate_loss` | a year of claims simulated by the compiled core: Poisson frequency, lognormal severity, VaR and TVaR out |
| `chain_ladder`, `mack_chain_ladder`, `bornhuetter_ferguson` | claims triangles (need NumPy; BF new in 0.5.0) |
| `verify` | the verification suite, as data |
| `riskpy_status`, `riskpy_doctor`, `riskpy_explain` | introspection: recent calls and error counts, the installation check, an error code or a tool's formula and conventions — *new in 0.5.0* |

Every tool is a thin wrapper over a public function that [the verification suite](verification.md)
checks. Arguments are plain numbers, strings and lists: rates as decimals, times in years,
probability levels as decimals. Results are JSON. The wrappers hide the conventions that
bite — `aggregate_loss` takes a severity mean and standard deviation on the money scale
and converts to the lognormal parameters itself, so an agent cannot make the commonest
severity mistake there is. A tool whose extra is not installed is reported as
`available: false` with the `pip install` line, and calling it raises `ToolUnavailable`
(code RP201) saying the same.

Arguments are checked before anything runs, on every path (`tools.call`, the CLI, MCP, a
direct Python call): numbers must be finite numbers (no `NaN`, `Infinity`, `true` or `"100"`),
whole numbers whole, and the work one call can ask for is bounded — `aggregate_loss` takes at
most 1,000,000 trials and 10⁸ simulated claims (trials × expected frequency),
`bond_analytics` and `cds_par_spread` at most 100,000 payment periods (maturity × frequency),
the triangle tools at most 300 × 300. The limits are in `riskpy.tools.LIMITS`; a call over one
fails at once with code RP405 and a message naming it. The tools also refuse percentages where
decimals belong — a rate above 100% (RP104), a volatility above 500% (RP105), an LGD above 1
(RP103), an expected loss ratio above 500% (RP104) — because the library would compute a confident wrong answer from them
(`riskpy.tools.PLAUSIBLE`). Results never contain `NaN` or `Infinity` (they become `null`), so
they are strict JSON.

## When a call fails

Every refusal carries a stable code, the parameter and a hint. `tools.call(name, args,
errors="return")`, `riskpy-tools call` and `riskpy-mcp` all return it as
`{"error": {"code", "message", "param", "hint", "docs"}}` — no tracebacks — and the default
`errors="raise"` raises it as the `ValueError` / `TypeError` / `KeyError` earlier versions raised,
with `.code`, `.param`, `.hint` and `.to_dict()`. An unknown argument gets a suggestion (`vol` →
`sigma`), a misspelt tool a "did you mean". The [error codes](errors.md) page lists them all;
[Diagnostics](diagnostics.md) covers `riskpy-doctor` and the call trace.

The manifest's `input_schema` is JSON Schema, so the same list drives Anthropic tool use,
OpenAI function calling, or any framework that takes schemas.

## Over MCP

```bash
pip install "open-riskpy[mcp]"
riskpy-mcp --config                  # the client configuration, and the `claude mcp add` line
claude mcp add riskpy -- riskpy-mcp  # Claude Code
```

`riskpy-mcp` serves every available tool on stdio under its own name, with the schema
derived from its signature (every parameter described) and the description from its
docstring, so Claude Code, Claude Desktop, Cursor and any other MCP client can price an
option, size Basel capital or simulate a year of claims with numbers the verification suite
stands behind. Calls run through `tools.call`, so a string sent where a number belongs is
refused (RP404) rather than silently coerced, and every failure is a JSON error object the
agent can act on. `riskpy_status`, `riskpy_doctor` and `riskpy_explain` let it look at the
recent calls, the installation and the meaning of a code without leaving the session.

## The context pack

```bash
riskpy-context                       # Markdown: install, conventions that bite, every module's
                                     # public functions with signatures, the tools, verification
riskpy-context --json                # the same as data
riskpy-context -o AGENTS.md          # write it for the agent working in your project
```

It is generated from the installed package, so the signatures are the ones that will run.
The "conventions that bite" section is the part to read first: rates are decimals, `i` is
effective annual while `r` is continuous, `LogNormal(mu, sigma)` takes the underlying
normal's parameters, seeds make runs reproducible, and `riskpy-verify` is how to check
rather than trust.
