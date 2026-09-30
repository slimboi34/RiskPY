# For AI agents

An agent that can call functions needs three things from a library: a list of what it may
call, a schema for each call, and results it can pass on as data. RiskPY 0.4.0 provides all
three, generated from the calculators' own signatures and docstrings, plus a description of
the library written for the model that is about to use it.

```bash
pip install open-riskpy              # the core: sixteen tools run with no dependencies
pip install "open-riskpy[sim]"       # + NumPy: the claims-triangle tools
pip install "open-riskpy[mcp]"       # + the MCP SDK: riskpy-mcp
```

## The tools

```bash
riskpy-tools                         # the list
riskpy-tools --json                  # the manifest: name, description, input schema, what it needs
riskpy-tools call black_scholes '{"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}'   # 10.4506
riskpy-tools schema aggregate_loss
```

```python
from riskpy import tools

tools.manifest()                                     # a list of dicts, one per tool
tools.call("basel_irb_capital", {"pd": 0.01, "lgd": 0.45, "ead": 1e6})["risk_weight"]   # 0.9232
```

| tool | what it does |
|---|---|
| `black_scholes`, `option_greeks`, `implied_vol` | European options, pure Python |
| `bond_analytics` | price, Macaulay and modified duration, convexity, DV01 at a yield |
| `basel_irb_capital`, `expected_loss`, `merton`, `cds_par_spread` | credit risk |
| `annuity_certain`, `life_annuity_due`, `life_insurance`, `net_premium_reserve` | life contingencies on the AMLCR standard table |
| `aggregate_loss` | a year of claims simulated by the compiled core: Poisson frequency, lognormal severity, VaR and TVaR out |
| `chain_ladder`, `mack_chain_ladder` | claims triangles (need NumPy) |
| `verify` | the verification suite, as data |

Every tool is a thin wrapper over a public function that [the verification suite](verification.md)
checks. Arguments are plain numbers, strings and lists: rates as decimals, times in years,
probability levels as decimals. Results are JSON. The wrappers hide the conventions that
bite — `aggregate_loss` takes a severity mean and standard deviation on the money scale
and converts to the lognormal parameters itself, so an agent cannot make the commonest
severity mistake there is. A tool whose extra is not installed is reported as
`available: false` with the `pip install` line, and calling it raises `ToolUnavailable`
saying the same.

The manifest's `input_schema` is JSON Schema, so the same list drives Anthropic tool use,
OpenAI function calling, or any framework that takes schemas.

## Over MCP

```bash
pip install "open-riskpy[mcp]"
riskpy-mcp --config                  # the client configuration, and the `claude mcp add` line
claude mcp add riskpy -- riskpy-mcp  # Claude Code
```

`riskpy-mcp` serves every available tool on stdio under its own name, with the schema
derived from its signature and the description from its docstring, so Claude Code, Claude
Desktop, Cursor and any other MCP client can price an option, size Basel capital or
simulate a year of claims with numbers the verification suite stands behind.

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
