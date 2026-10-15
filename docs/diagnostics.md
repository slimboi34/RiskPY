# Diagnostics

Three things answer "what went wrong, and what is going on?" — for a person at a terminal and
for an agent calling RiskPY's tools.

| question | person | Python | agent (MCP) |
|---|---|---|---|
| Is this installation healthy? | `riskpy-doctor` | `riskpy.doctor()` | `riskpy_doctor` |
| What does this error mean? | `riskpy-explain RP105` | `riskpy.errors.explain("RP105")` | `riskpy_explain` |
| What has been called, and what failed? | `riskpy-tools trace` | `riskpy.tools.trace()` | `riskpy_status` |

## riskpy-doctor

```bash
riskpy-doctor            # a table: each check ok / warn / fail, with the fix
riskpy-doctor --json     # {"ok", "version", "summary", "checks": [{id, status, message, hint}, ...]}
```

```text
riskpy-doctor — RiskPY 0.5.1

  ok    python            Python 3.12.13 (CPython) at /…/bin/python
  ok    riskpy_version    open-riskpy 0.5.1 (riskpy at /…/site-packages/riskpy)
  ok    core_import       compiled core loaded from /…/riskpy/cpp_underwriter.cpython-312-darwin.so
  ok    core_compute      the core simulates correctly (Poisson mean 3.0064 ≈ 3)
  ok    core_hardening    the core refuses non-finite parameters (0.4.1+ build)
  ok    extra_numpy       numpy 2.5.3
  warn  extra_pandas      pandas is not installed — needed for the .to_frame() methods
                          → pip install pandas
  …
  ok    selfcheck         68 identities verified in 0.16s (core, special, quant, life, rates, credit)
  ok    tools             27 of 27 calculators available
  ok    mcp_server        the MCP server builds and lists 30 tools; command /…/bin/riskpy-mcp
  ok    platform          Darwin 25.1.0 arm64

14 ok, 1 warn, 0 fail — works, with warnings
```

| check | what it looks at |
|---|---|
| `python` | the interpreter; RiskPY needs 3.10+ and ships wheels for 3.10–3.14 |
| `riskpy_version` | the installed `open-riskpy` distribution, and whether the `riskpy` being imported *is* that installation — a source checkout on `sys.path` shadowing it is the commonest confusion |
| `core_import` | the compiled C++ core imports, from the same directory as the Python layer |
| `core_compute` | the core answers a known question (a Poisson count's mean) correctly |
| `core_hardening` | the core refuses `NaN` parameters, which only 0.4.1+ builds do — an old extension under a new Python layer shows up here |
| `extra_*` | NumPy, SciPy, pandas, Matplotlib, the MCP SDK, Tkinter: installed or not, with the exact `pip` line |
| `selfcheck` | a subset of `riskpy-verify` (core, special functions, quant, life, rates, credit) |
| `tools` | the agent tools load; which need an extra |
| `mcp_server` | the MCP server builds, how many tools it lists, and where `riskpy-mcp` is |
| `platform` | the OS and architecture, for bug reports |

A missing optional extra is a **warn**; a broken core, a failing identity or an MCP server that
will not build is a **fail**, and the command exits 1 — so `riskpy-doctor` can gate a CI job or a
deployment.

## Error codes

Every deliberate error has a stable code (`RP105`), the parameter at fault and a hint. The full
list, with meanings and fixes, is on the [error codes](errors.md) page; `riskpy-explain CODE`
prints one entry.

Over the tools, a failed call is data, not a traceback:

```python
from riskpy import tools

tools.call("black_scholes", {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 20}, errors="return")
# {"error": {"code": "RP105", "param": "sigma",
#            "message": "sigma = 20 looks like a percentage: volatility is a decimal, so sigma must be at most 5 (500%)",
#            "hint": "Divide by 100 — 20% is 0.2.", "docs": "…/errors/#rp105"}}
```

`riskpy-tools call` prints the same object on stdout (exit status 1) and a one-line summary on
stderr; `riskpy-mcp` returns it as the text of a tool error. The default `errors="raise"` raises
the error instead, as the same `ValueError` / `TypeError` / `KeyError` earlier versions raised.

### Percentages where decimals belong

The library computes with whatever numbers it is given: `black_scholes(r=5)` is a 500% rate and
returns a price. An agent that meant 5% gets a confident wrong answer. The *tools* therefore
refuse values no market quotes (`riskpy.tools.PLAUSIBLE`):

| parameters | refused when | code |
|---|---|---|
| `r`, `i`, `q`, `ytm`, `coupon`, `rate`, `spread`, `theta`, `par_rates` | above 1 (100%) in absolute value | RP104 |
| `sigma` | negative, or above 5 (500% a year) | RP105 |
| `lgd` | outside [0, 1] | RP103 |
| `expected_loss_ratio` | above 5 (500%) | RP104 |

The library functions underneath are unchanged.

## The trace

The tool layer remembers its recent calls — tool, arguments (long lists abbreviated), duration,
and the error code if the call failed:

```python
tools.trace()             # the last 200 calls in this process, oldest first
tools.status()            # version, tools, limits, call and error counts by code, recent calls
```

```bash
riskpy-tools trace                       # this process's calls, plus RISKPY_TRACE_FILE if set
RISKPY_TRACE_FILE=~/riskpy-trace.jsonl riskpy-tools call black_scholes '{…}'
riskpy-tools trace --json --limit 20
```

The buffer lives in the process that made the calls, so over MCP the agent asks the server:
`riskpy_status` returns the counts and the recent calls, each with `via: "mcp"`. Set
`RISKPY_TRACE_FILE` to append every call to a JSON-lines file as well, which is how separate
`riskpy-tools call` processes share one trace.

## Explaining a tool

`riskpy_explain` takes an error code *or* a tool name. For a tool it returns the schema plus the
formula and conventions it uses — for `black_scholes`, that `r` and `q` are continuously
compounded and `T = 0` gives the discounted intrinsic value. `black_scholes` and
`bond_analytics` also take `explain: true` and add the working to the result (for Black–Scholes,
`d1`, `d2`, `N(d1)`, `N(d2)` and the discounted spot and strike).

## Measured

`benchmarks/diagnostics/` sends 19 realistic agent mistakes — a negative volatility, `r=5`, a
maturity of 0, a probability above 1, a misspelt tool, `vol` for `sigma`, a number sent as a
string, an impossible option price, a missing extra — over MCP and the CLI, and scores each reply
for a code, the parameter, a hint and structured JSON: 3 of 76 points for 0.4.0, 13 for the
unpublished 0.4.1, 76 for 0.5.0. Six of the nineteen mistakes returned a number before 0.5.0 —
a confident wrong answer. The table, the replies the agent actually saw and the caveats are in
[`benchmarks/diagnostics/RESULTS.md`](https://github.com/slimboi34/RiskPY/blob/main/benchmarks/diagnostics/RESULTS.md).
