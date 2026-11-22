# For AI agents

An agent that can call functions needs three things from a library: a list of what it may
call, a schema for each call, and results it can pass on as data. RiskPY provides all three,
generated from the calculators' own signatures and docstrings, plus a description of the
library written for the model that is about to use it. Since 0.5.0 it also tells the agent
exactly what went wrong when a call fails — a stable error code, the parameter, the fix, as
JSON — and lets it look at what has been happening ([Diagnostics](diagnostics.md)).
Since 0.7.0 it can drive any calculator through a stress test with one call, the MCP server
offers resources and prompts as well as tools, and every call can be recorded to a
tamper-evident ledger ([The audit ledger](audit.md)).

```bash
pip install open-riskpy              # the core: most tools run with no dependencies
pip install "open-riskpy[sim]"       # + NumPy: the claims-triangle, VaR and capital-allocation tools
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
riskpy-audit replay run.jsonl        # recompute every call in an audit ledger (RISKPY_AUDIT_FILE writes one)
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
| `black_scholes`, `option_greeks`, `implied_vol` | European options, pure Python (`black_scholes` takes `explain: true` for d1, d2 and the convention; `option_greeks` gives vega and rho per percentage point and theta per calendar day) |
| `value_at_risk` | historical or parametric VaR and expected shortfall of a return series (NumPy) — *new in 0.5.0* |
| `bond_analytics` | price, Macaulay and modified duration, convexity, DV01 at a yield |
| `convert_rate`, `bootstrap_yield_curve`, `short_rate_zero_bond` | compounding conventions, a zero curve from par yields, Vasicek / CIR bond prices — *new in 0.5.0* |
| `basel_irb_capital`, `expected_loss`, `merton`, `cds_par_spread` | credit risk |
| `hazard_from_spread`, `vasicek_loss_quantile` | the credit triangle; the ASRF portfolio loss quantile — *new in 0.5.0* |
| `annuity_certain`, `life_annuity_due`, `life_insurance`, `net_premium_reserve` | life contingencies on the AMLCR standard table |
| `term_insurance`, `endowment_insurance`, `temporary_annuity_due`, `life_premium` | n-year contracts and net premiums — *new in 0.5.0* |
| `aggregate_loss` | a year of claims simulated by the compiled core: Poisson frequency, lognormal severity, VaR and TVaR out |
| `chain_ladder`, `mack_chain_ladder`, `bornhuetter_ferguson` | claims triangles (need NumPy; BF new in 0.5.0) |
| `capital_aggregate`, `capital_allocation` | diversified capital from standalone figures and a correlation matrix; simulated lines joined by a copula with TVaR or VaR allocated to them ([Economic capital](capital.md); allocation needs NumPy) — *new in 0.7.0* |
| `copula_dependence`, `nearest_correlation` | what a copula implies (Kendall's tau, tail dependence, joint exceedance); repair a correlation matrix that is not positive semi-definite (NumPy) — *new in 0.7.0* |
| `sensitivity`, `scenario_table` | bump-and-revalue any calculator: derivatives and elasticities of every output; any calculator under named what-ifs ([Stress and scenario testing](stress.md)) — *new in 0.7.0* |
| `verify` | the verification suite, as data |
| `riskpy_status`, `riskpy_doctor`, `riskpy_explain` | introspection: recent calls and error counts, the installation check, an error code or a tool's formula and conventions — *new in 0.5.0* |
| `riskpy_audit` | introspection: whether calls are being recorded to the audit ledger, the head digest to quote, and optionally a replay — *new in 0.7.0* |

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
the triangle tools at most 300 × 300, the capital tools at most 100 risks and 5 × 10⁶ simulated cells
(trials × lines: about 2 s and 250 MB at the limit), `scenario_table` at most 100 scenarios. Strings, tool names, argument names and
scenario labels are at most 200 characters, and an object argument nests at most 32 levels deep. The
limits are in `riskpy.tools.LIMITS`; a call over one fails at once with code RP405 and a message naming
it, and never echoes the oversized value back. The tools also refuse percentages where
decimals belong — a rate above 100% (RP104), a volatility above 500% (RP105), an LGD above 1
(RP103), an expected loss ratio above 500% (RP104) — because the library would compute a confident wrong answer from them
(`riskpy.tools.PLAUSIBLE`). Results never contain `NaN` or `Infinity`, so they are strict JSON: a
number that overflowed is refused with RP304, never returned as `null`. A `null` number is a value
that is undefined by construction, and a `note` beside it says why: `merton`'s
`distance_to_default` when σ√T is zero (with `"deterministic": true`), `mack_chain_ladder`'s
`total_cv` when the total reserve is 0, `black_scholes`' `explain.d1` / `d2` when σ√T is too small
for them to be finite.

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

### Tools that drive other tools

`sensitivity` and `scenario_table` take the name of another calculator and its arguments as an
object:

```python
from riskpy import tools

tools.call("sensitivity", {"tool": "bond_analytics", "parameter": "ytm",
                           "arguments": {"face": 100, "coupon": 0.05, "maturity": 10, "ytm": 0.04}})
# outputs.price: base 108.1757, derivative −857.02 per 1.00 of yield (−0.0857 per basis point),
#                second_derivative 8164.3
```

The inner call is checked exactly as a direct call is, so a percentage where a decimal belongs
is still refused with the same code. An agent no longer needs three calls and its own
arithmetic to get a duration or a delta, and cannot get the finite difference wrong.

One meta-tool call is still one tool call. `sensitivity` runs the calculator up to four times and
`scenario_table` once per scenario plus the base, and all those runs together may ask for no more
work than one call of the calculator at its limit: 100 scenarios of a one-million-trial
simulation are refused before anything runs (RP405), with a message naming the largest run.
`verify` cannot be driven. A scenario table still running after `LIMITS["meta_seconds"]` (20 s)
returns the scenarios it finished and lists the rest under `not_run`, each with RP405. Over MCP, a
client that cancels the request stops the work at the next gap between runs (RP408 in the trace).

For a simulation (`aggregate_loss`, `capital_allocation`) the default bump is 1% of the value —
1% of its distance to 0 or 1 for a confidence level — rather than the 0.0001 × max(1, |value|) a
closed form gets: a smaller bump measures sampling noise (a 0.01% bump of the claim frequency gave
a derivative of −12,566 where the true one is 1,000). A simulated derivative still carries some
noise; repeat it with a different bump or more trials before relying on it.

### Resources and prompts

Besides tools, `riskpy-mcp` serves reference text as MCP **resources**, which a client can
attach to the conversation without a tool call:

| resource | what it is |
|---|---|
| `riskpy://context` | the context pack (Markdown) |
| `riskpy://conventions` | the conventions that bite (Markdown) |
| `riskpy://errors` | the error catalogue (JSON) |
| `riskpy://tools` | the tool manifest (JSON) |
| `riskpy://audit` | the audit ledger's status and head digest (JSON) |
| `riskpy://explain/{topic}` | an error code or a tool explained (JSON) |

and **prompts**, which are step-by-step procedures for jobs that take several tools in the
right order:

| prompt | arguments | what it walks through |
|---|---|---|
| `capital_model` | `lines`, `level` | repair the correlation matrix, aggregate, allocate with Euler contributions, compare copulas |
| `reserve_review` | `triangle` | chain ladder, Mack standard errors, Bornhuetter–Ferguson, sensitivity to the tail factor |
| `stress_test` | `tool`, `base_case` | sensitivities to each input, named scenarios, where the linear estimate breaks down |
| `audit_trail` | — | the ledger head, the chain check and the replay for this conversation's numbers |

Every prompt ends by asking for the ledger head. The same text is available without MCP:

```python
from riskpy import mcp_server

mcp_server.read_resource("riskpy://conventions")
mcp_server.render_prompt("capital_model", lines="motor, property, liability", level="0.995")
```

### Recording a session

```bash
riskpy-mcp --audit session.jsonl     # or set RISKPY_AUDIT_FILE in the client's server configuration
```

Every calculator call the agent makes is then written in full to a hash-chained ledger.
`riskpy_audit` gives the agent the head digest to quote, and `riskpy-audit replay
session.jsonl` lets anyone recompute the session afterwards. See [The audit ledger](audit.md).
The server opens the ledger when it starts and exits with an RP505 message if it cannot (a
mistyped folder would otherwise record nothing and say nothing); `riskpy-tools call` warns on
stderr when a call could not be recorded.

## The context pack

```bash
riskpy-context                       # Markdown: install, conventions that bite, every module's
                                     # public functions with signatures, the tools, verification
riskpy-context --json                # the same as data
riskpy-context -o AGENTS.md          # write it for the agent working in your project
```

It is generated from the installed package, so the signatures are the ones that will run:
functions and class constructors with their signatures, each class's own summary (not one
inherited from a base class) and all its public methods, and a worked example of a model run
with a copula, allocated and stressed.
The "conventions that bite" section is the part to read first: rates are decimals, `i` is
effective annual while `r` is continuous, `LogNormal(mu, sigma)` takes the underlying
normal's parameters, seeds make runs reproducible, a copula's correlation is not a Spearman
correlation, losses are positive, and `riskpy-verify` is how to check rather than trust.
