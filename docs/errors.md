# Error codes

Every error RiskPY raises on purpose has a **stable code** that never changes meaning between
releases, names the **parameter** at fault, and carries a **hint** that says how to fix the call.
The exceptions are still the built-in types RiskPY always raised — a negative volatility is a
`ValueError`, an unknown tool a `KeyError` — so existing `except` clauses keep working.

```python
from riskpy import quant
from riskpy.errors import RiskPyError

try:
    quant.black_scholes(S=100, K=100, T=1, r=0.05, sigma=-0.2)
except RiskPyError as exc:
    exc.code       # "RP105"
    exc.param      # "sigma"
    exc.hint       # "sigma is the annual volatility as a decimal (0.2 means 20%) and cannot be negative."
    exc.to_dict()  # {"code", "message", "param", "hint", "docs"}
```

From the command line, and for an agent:

```bash
riskpy-explain RP105            # what the code means and how to fix it
riskpy-explain --list           # every code, by family
riskpy-explain --list --json    # the catalogue as data
```

Over the agent tools (`tools.call(..., errors="return")`, `riskpy-tools call`, `riskpy-mcp`) a
failed call comes back as JSON instead of a traceback:

```json
{"error": {"code": "RP104", "message": "r = 5 looks like a percentage: rates are decimals, so |r| must be at most 1",
           "param": "r", "hint": "Divide by 100 — 5% is 0.05.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp104"}}
```

| family | meaning |
|---|---|
| `RP1xx` | an input outside its domain |
| `RP2xx` | an optional dependency that is not installed, or a compiled core that will not load |
| `RP3xx` | a numerical problem: no solution exists, or a solver cannot reach one |
| `RP4xx` | a tool call that is wrong as a call (unknown tool or argument, wrong JSON type, over a limit) |
| `RP5xx` | data that is malformed as a whole (a life table, a claims triangle, a curve, a matrix) |
| `RP9xx` | something RiskPY did not expect — a bug worth reporting |

The catalogue lives in `riskpy/errors.py`; a test checks that every code the package raises is in it
and on this page. A code is never reused for a different meaning.

## RP1xx — input domain

### RP100 — Invalid input {#rp100}

An argument is outside what the function accepts. The message names it and the allowed range.

**Fix:** Read the message: it names the parameter and the values it accepts.

### RP101 — Not a finite number {#rp101}

A numeric argument is NaN, infinite, or not a number at all (a string, None).

**Fix:** Pass a finite number. JSON NaN/Infinity and numeric strings such as "100" are refused, not coerced.

### RP102 — Must be positive (or non-negative) {#rp102}

A quantity that cannot be negative or zero (a price, a strike, a face value, a mean) was.

**Fix:** Check the sign and units; prices, strikes, exposures and sizes are positive amounts of money.

### RP103 — Probability outside [0, 1] {#rp103}

A probability (PD, LGD, recovery, a weight) is below 0 or above 1. Above 1 is almost always a percentage passed where a decimal belongs.

**Fix:** Probabilities are decimals: 1% is 0.01, 45% is 0.45.

### RP104 — Rate looks like a percentage {#rp104}

An interest rate, yield, coupon, dividend yield or spread is larger than 1 (100%) in absolute value, or an expected loss ratio larger than 5 (500%). RiskPY takes rates and ratios as decimals, so 5 means 500%, not 5%.

**Fix:** Divide by 100: a 5% rate is 0.05.

### RP105 — Volatility out of range {#rp105}

A volatility is negative, or so large (above 5, i.e. 500% a year) that it is almost certainly a percentage passed where a decimal belongs.

**Fix:** Volatility is an annual decimal: 0.2 means 20%. It must be >= 0.

### RP106 — Time or maturity out of range {#rp106}

A time to expiry, maturity, duration or horizon is negative, zero where a positive time is needed, or past the end of the contract.

**Fix:** Times are in years (6 months is 0.5). Implied volatility and similar inversions need T > 0.

### RP107 — Whole number required {#rp107}

An age, term, count or payment frequency must be a whole number (and usually at least 1); a fraction or a boolean was given.

**Fix:** Pass an integer: ages and terms are whole years, frequency is payments per year (1, 2, 4, 12).

### RP108 — Unknown choice {#rp108}

A string option is not one of the accepted values (option kind, asset class, contract kind, compounding convention, life table).

**Fix:** Use one of the values the message lists; they are case-insensitive where noted.

### RP109 — Lengths do not match {#rp109}

Sequences that pair up element by element (times and discount factors, par rates and maturities, PDs, LGDs and EADs, premiums and origin periods) have different lengths.

**Fix:** Give every list one entry per item; the message says which lengths differ.

### RP110 — Confidence level outside (0, 1) {#rp110}

A quantile or confidence level must lie strictly between 0 and 1.

**Fix:** Levels are decimals: the 99.5th percentile is 0.995, not 99.5.

### RP111 — Empty input {#rp111}

A list that needs at least one (sometimes two) entries is empty.

**Fix:** Pass the data; the message says the minimum length.

### RP112 — Not in increasing order {#rp112}

Times, maturities or ages must be strictly increasing and were not.

**Fix:** Sort the knots by time and remove duplicates.

### RP113 — Inconsistent arguments {#rp113}

Arguments that are each valid do not fit together (a term given for a whole-life contract, a valuation time after expiry, t_1 >= t_2, an age outside the table).

**Fix:** Read the message: it names both arguments and the relation they must satisfy.


## RP2xx — missing dependency

### RP201 — NumPy not installed {#rp201}

The simulation layer (riskpy.mc, riskpy.reserving, path simulators, portfolio simulation, the chain-ladder tools) needs NumPy, which is an optional extra.

**Fix:** pip install "open-riskpy[sim]"

### RP202 — Matplotlib not installed {#rp202}

riskpy.viz draws with Matplotlib, which is an optional extra.

**Fix:** pip install "open-riskpy[viz]"

### RP203 — pandas not installed {#rp203}

The .to_frame() methods return pandas DataFrames; pandas is not a dependency.

**Fix:** pip install pandas

### RP204 — MCP SDK not installed {#rp204}

riskpy-mcp serves the tools over the Model Context Protocol and needs the MCP SDK (Python >= 3.10).

**Fix:** pip install "open-riskpy[mcp]"

### RP205 — Compiled core not importable {#rp205}

The C++ extension cpp_underwriter did not load: usually a source checkout without a build, or a wheel for a different Python or platform.

**Fix:** pip install --force-reinstall open-riskpy  (or build a checkout with `pip install -e .`); run riskpy-doctor for details.

### RP206 — Tkinter not available {#rp206}

The optional desktop UnderwritingApp needs Tkinter, which some Python builds omit.

**Fix:** Install a Python with Tk support, or use the headless APIs.


## RP3xx — numerical

### RP301 — Price below the no-arbitrage floor {#rp301}

No volatility (or spread, or yield) reproduces the price: it is below the option's discounted intrinsic value. The quote is stale or the inputs (S, K, T, r, q, kind) are wrong.

**Fix:** Check the price and that kind matches the quote (a put priced as a call is the usual slip).

### RP302 — Price above the no-arbitrage ceiling {#rp302}

The price exceeds what any volatility can produce (a call is worth at most S·e^(−qT), a put at most K·e^(−rT)), or what the solver's widest bracket reaches.

**Fix:** Check the price units (per share, not per contract of 100) and the spot and strike.

### RP303 — Root not bracketed {#rp303}

A solver (yield to maturity, z-spread, implied hazard) found no sign change on its search interval, so no solution exists in the range it searched.

**Fix:** Check that the target price is attainable for the instrument; extreme prices have no solution.

### RP304 — Numerical overflow {#rp304}

An intermediate result overflowed floating point (for example (1 + i)^n with a large n and a rate near -1).

**Fix:** Use inputs of a realistic size; the message names the calculation that overflowed.

### RP305 — Non-finite result {#rp305}

A calculation produced NaN or infinity: a simulation formula divided by a variable that reached zero, or a resampled triangle had a degenerate column.

**Fix:** Look for a division by something that can be zero, or a log of something that can go negative.


## RP4xx — tool call

### RP401 — Unknown tool {#rp401}

No tool has that name.

**Fix:** List the tools with `riskpy-tools` (or tools.manifest(), or the MCP tools/list request).

### RP402 — Unexpected argument {#rp402}

The call passed an argument the tool does not take (often a misspelling, or a name from another library such as `vol` for `sigma`).

**Fix:** Use only the argument names in the tool's input_schema: `riskpy-tools schema NAME`.

### RP403 — Missing required argument {#rp403}

A required argument was not passed.

**Fix:** Pass every name in the schema's `required` list: `riskpy-tools schema NAME`.

### RP404 — Wrong argument type {#rp404}

An argument has the wrong JSON type: a string where a number belongs, true/false where a number belongs, a scalar where a list belongs.

**Fix:** Send JSON numbers as numbers (100, not "100"), booleans as true/false, lists as arrays.

### RP405 — Over a tool limit {#rp405}

The call asks for more work than one tool call may (trials, simulated claims, payment periods, triangle size, list or string length, integer size). The limits are in riskpy.tools.LIMITS.

**Fix:** Lower the size named in the message, or call the library function directly, which is unbounded.

### RP406 — Arguments are not a JSON object {#rp406}

The arguments for a tool call could not be parsed as a JSON object.

**Fix:** Pass a JSON object, e.g. riskpy-tools call black_scholes '{"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2}'.

### RP407 — Unknown error code {#rp407}

riskpy-explain was asked about a code that is not in the catalogue.

**Fix:** riskpy-explain --list shows every code.


## RP5xx — data

### RP501 — Invalid life table {#rp501}

Mortality data is malformed: q_x outside [0, 1], l_x negative or increasing, too few ages, or a final age whose q is not 1.

**Fix:** Check the table's columns and that the last q_x is 1 (everyone dies by the end of the table).

### RP502 — Invalid claims triangle {#rp502}

A claims triangle is malformed: ragged rows, too few origin or development periods, non-finite cumulative amounts, or no observed link ratios.

**Fix:** Rows are origin periods, columns development periods, cumulative amounts, null where unobserved.

### RP503 — Invalid term structure {#rp503}

A yield curve, hazard curve or discount-factor set is malformed: no knots, non-positive discount factors, or (time, rate) pairs that are not pairs.

**Fix:** Give strictly increasing positive times with one rate or discount factor each.

### RP504 — Invalid transition matrix {#rp504}

A rating transition matrix is not square, has rows that do not sum to 1, negative entries, duplicate ratings, or a default state that is not absorbing.

**Fix:** Each row is a probability distribution over the next rating: non-negative, summing to 1.

### RP505 — File or I/O error {#rp505}

A file could not be read or written (an export, a context pack, a trace file).

**Fix:** Check the path exists and is writable.


## RP9xx — internal

### RP900 — Unexpected internal error {#rp900}

RiskPY raised something it did not expect. The call may have been valid; this is a bug.

**Fix:** Run riskpy-doctor, then report the call and the message at https://github.com/slimboi34/RiskPY/issues.

