# Diagnostics benchmark: what an agent sees when it gets a call wrong

19 realistic agent mistakes (`mistakes.py`), each sent over MCP (`riskpy-mcp` on stdio) and the CLI (`riskpy-tools call`) against an installed RiskPY. Scored by `run.py` — the rules are in its docstring: **(a)** a stable error code, **(b)** the offending parameter named, **(c)** an actionable hint, **(d)** structured JSON on both channels with no traceback. *Detected* counts mistakes that raised any error at all; an undetected mistake returned a number — a confident wrong answer — and scores zero.

## Summary

| version | detected | (a) code | (b) param | (c) hint | (d) JSON | score |
|---|---|---|---|---|---|---|
| 0.4.0 (PyPI) (`0.4.0`) | 13/19 | 0/19 | 3/19 | 0/19 | 0/19 | **3/76** (4%) |
| 0.4.1 (main, unreleased) (`0.4.0`) | 13/19 | 0/19 | 10/19 | 3/19 | 0/19 | **13/76** (17%) |
| 0.5.0 (`0.5.0`) | 19/19 | 19/19 | 19/19 | 19/19 | 19/19 | **76/76** (100%) |

## Per mistake

Each cell is detected · code · param · hint · JSON (✓ met, · not met).

| mistake | 0.4.0 (PyPI) | 0.4.1 (main, unreleased) | 0.5.0 |
|---|---|---|---|
| `negative_vol` — a negative volatility | ✓ · · · · | ✓ · ✓ · · | ✓ ✓ ✓ ✓ ✓ |
| `vol_as_percent` — volatility given as a percentage (20 for 20%) | · · · · · | · · · · · | ✓ ✓ ✓ ✓ ✓ |
| `rate_as_percent` — interest rate given as a percentage (5 for 5%) | · · · · · | · · · · · | ✓ ✓ ✓ ✓ ✓ |
| `coupon_as_percent` — bond coupon given as a percentage | · · · · · | · · · · · | ✓ ✓ ✓ ✓ ✓ |
| `maturity_zero` — a bond with maturity 0 | ✓ · · · · | ✓ · ✓ · · | ✓ ✓ ✓ ✓ ✓ |
| `implied_vol_at_expiry` — implied volatility asked for at expiry (T = 0) | ✓ · · · · | ✓ · · · · | ✓ ✓ ✓ ✓ ✓ |
| `probability_above_1` — a default probability above 1 | ✓ · · · · | ✓ · ✓ ✓ · | ✓ ✓ ✓ ✓ ✓ |
| `lgd_as_percent` — loss given default as a percentage (45 for 45%) | · · · · · | · · · · · | ✓ ✓ ✓ ✓ ✓ |
| `recovery_as_percent` — recovery rate as a percentage | ✓ · · · · | ✓ · ✓ ✓ · | ✓ ✓ ✓ ✓ ✓ |
| `implied_vol_impossible_price` — a call priced above the spot (no volatility can produce it) | ✓ · · · · | ✓ · ✓ ✓ · | ✓ ✓ ✓ ✓ ✓ |
| `implied_vol_below_intrinsic` — a call priced below its intrinsic value | ✓ · · · · | ✓ · ✓ · · | ✓ ✓ ✓ ✓ ✓ |
| `unknown_tool` — a tool name that does not exist (misspelt) | ✓ · ✓ · · | ✓ · ✓ · · | ✓ ✓ ✓ ✓ ✓ |
| `wrong_argument_name` — another library's argument name (vol for sigma) | ✓ · ✓ · · | ✓ · ✓ · · | ✓ ✓ ✓ ✓ ✓ |
| `missing_argument` — a required argument left out | ✓ · ✓ · · | ✓ · ✓ · · | ✓ ✓ ✓ ✓ ✓ |
| `number_as_string` — a number sent as a JSON string | · · · · · | · · · · · | ✓ ✓ ✓ ✓ ✓ |
| `unknown_choice` — an asset class spelt as the 0.4.0 tool description suggested | ✓ · · · · | ✓ · ✓ · · | ✓ ✓ ✓ ✓ ✓ |
| `ragged_triangle` — a claims triangle whose rows have mismatched lengths | · · · · · | · · · · · | ✓ ✓ ✓ ✓ ✓ |
| `age_outside_table` — an age below the life table's first age | ✓ · · · · | ✓ · · · · | ✓ ✓ ✓ ✓ ✓ |
| `missing_extra` — a tool whose optional dependency (NumPy) is not installed | ✓ · · · · | ✓ · · · · | ✓ ✓ ✓ ✓ ✓ |

## What the agent actually saw over MCP

### `negative_vol`

- **0.4.0 (PyPI)**: `Error executing tool black_scholes`
- **0.4.1 (main, unreleased)**: `Error executing tool black_scholes: sigma must be >= 0, got -0.2`
- **0.5.0**: `{ "error": { "code": "RP105", "message": "sigma must be >= 0, got -0.2", "param": "sigma", "hint": "sigma is an annual volatility as a decimal (0.2 means 20%) and cannot be negative.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp105" } }`

### `vol_as_percent`

- **0.4.0 (PyPI)**: `100.0`
- **0.4.1 (main, unreleased)**: `100.0`
- **0.5.0**: `{ "error": { "code": "RP105", "message": "sigma = 20 looks like a percentage: volatility is a decimal, so sigma must be at most 5 (500%)", "param": "sigma", "hint": "Divide by 100 — 20% is 0.2.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp105" } }`

### `rate_as_percent`

- **0.4.0 (PyPI)**: `99.32620530009146`
- **0.4.1 (main, unreleased)**: `99.32620530009146`
- **0.5.0**: `{ "error": { "code": "RP104", "message": "r = 5 looks like a percentage: rates are decimals, so |r| must be at most 1", "param": "r", "hint": "Divide by 100 — 5% is 0.05.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp104" } }`

### `coupon_as_percent`

- **0.4.0 (PyPI)**: `{ "price": 4055.009739661535, "macaulay_duration": 4.962193081622883, "modified_duration": 4.853000568824335, "convexity": 34.062475696678895, "dv01": 1.967896457316565 }`
- **0.4.1 (main, unreleased)**: `{ "price": 4055.009739661535, "macaulay_duration": 4.962193081622883, "modified_duration": 4.853000568824335, "convexity": 34.062475696678895, "dv01": 1.967896457316565 }`
- **0.5.0**: `{ "error": { "code": "RP104", "message": "coupon = 5 looks like a percentage: rates are decimals, so |coupon| must be at most 1", "param": "coupon", "hint": "Divide by 100 — 5% is 0.05.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp104" } }`

### `maturity_zero`

- **0.4.0 (PyPI)**: `Error executing tool bond_analytics`
- **0.4.1 (main, unreleased)**: `Error executing tool bond_analytics: maturity must be > 0, got 0.0`
- **0.5.0**: `{ "error": { "code": "RP106", "message": "maturity must be > 0, got 0.0", "param": "maturity", "hint": "Times are in years (6 months is 0.5); a maturity or time to expiry must be positive.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp106" } }`

### `implied_vol_at_expiry`

- **0.4.0 (PyPI)**: `Error executing tool implied_vol`
- **0.4.1 (main, unreleased)**: `Error executing tool implied_vol: implied vol is undefined at or after expiry`
- **0.5.0**: `{ "error": { "code": "RP106", "message": "implied vol is undefined at or after expiry: T must be > 0, got 0", "param": "T", "hint": "At expiry an option is worth its intrinsic value whatever the volatility, so there is nothing to invert. Pass the time to expiry in years (> 0).", "docs": "https://sli`

### `probability_above_1`

- **0.4.0 (PyPI)**: `Error executing tool expected_loss`
- **0.4.1 (main, unreleased)**: `Error executing tool expected_loss: pd must be in [0, 1], got 1.5 — probabilities here are decimals, so 1% is 0.01`
- **0.5.0**: `{ "error": { "code": "RP103", "message": "pd must be in [0, 1], got 1.5 — probabilities here are decimals, so 1% is 0.01", "param": "pd", "hint": "Probabilities are decimals: 1% is 0.01, 45% is 0.45.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp103" } }`

### `lgd_as_percent`

- **0.4.0 (PyPI)**: `{ "capital_requirement": 7.385344111364106, "capital": 7385344.111364107, "rwa": 92316801.39205134, "correlation": 0.192783679165516, "conditional_pd": 0.14027267845651578, "maturity_adjustment": 1.2598095009238282, "risk_weight": 92.31680139205133 }`
- **0.4.1 (main, unreleased)**: `{ "capital_requirement": 7.385344111364106, "capital": 7385344.111364107, "rwa": 92316801.39205134, "correlation": 0.192783679165516, "conditional_pd": 0.14027267845651578, "maturity_adjustment": 1.2598095009238282, "risk_weight": 92.31680139205133 }`
- **0.5.0**: `{ "error": { "code": "RP103", "message": "lgd must be in [0, 1], got 45", "param": "lgd", "hint": "lgd is a decimal fraction of the exposure: 45% is 0.45.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp103" } }`

### `recovery_as_percent`

- **0.4.0 (PyPI)**: `Error executing tool cds_par_spread`
- **0.4.1 (main, unreleased)**: `Error executing tool cds_par_spread: recovery must be in [0, 1], got 40.0 — probabilities here are decimals, so 1% is 0.01`
- **0.5.0**: `{ "error": { "code": "RP103", "message": "recovery must be in [0, 1], got 40.0 — probabilities here are decimals, so 1% is 0.01", "param": "recovery", "hint": "Probabilities are decimals: 1% is 0.01, 45% is 0.45.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp103" } }`

### `implied_vol_impossible_price`

- **0.4.0 (PyPI)**: `Error executing tool implied_vol`
- **0.4.1 (main, unreleased)**: `Error executing tool implied_vol: price 150.0 exceeds what even 512000% volatility produces — check the inputs.`
- **0.5.0**: `{ "error": { "code": "RP302", "message": "price 150 exceeds what even 512000% volatility produces — check the inputs.", "param": "price", "hint": "No volatility gives more than 100.000000 for this call (S·e^(−qT)). Check the price is per unit, and S and K.", "docs": "https://slimboi34.github.io/Risk`

### `implied_vol_below_intrinsic`

- **0.4.0 (PyPI)**: `Error executing tool implied_vol`
- **0.4.1 (main, unreleased)**: `Error executing tool implied_vol: price 1.0 is below intrinsic (52.438529) — no volatility can produce it, so the quote violates no-arbitrage.`
- **0.5.0**: `{ "error": { "code": "RP301", "message": "price 1 is below intrinsic (52.438529) — no volatility can produce it, so the quote violates no-arbitrage.", "param": "price", "hint": "The call's floor here is 52.438529 (its discounted intrinsic value). Check the price, and that kind='call' matches the quo`

### `unknown_tool`

- **0.4.0 (PyPI)**: `Unknown tool: blackscholes`
- **0.4.1 (main, unreleased)**: `Unknown tool: blackscholes`
- **0.5.0**: `{ "error": { "code": "RP401", "message": "unknown tool 'blackscholes'; `riskpy-tools` lists them", "param": "name", "hint": "Did you mean 'black_scholes'? `riskpy-tools` (or tools.manifest()) lists every tool.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp401" } }`

### `wrong_argument_name`

- **0.4.0 (PyPI)**: `Error executing tool black_scholes: 1 validation error for black_scholesArguments sigma Field required [type=missing, input_value={'S': 100, 'K': 100, 'T':..., 'r': 0.05, 'vol': 0.2}, input_type=dict] For further information visit https://errors.pydantic.dev/2.14/v/missing`
- **0.4.1 (main, unreleased)**: `Error executing tool black_scholes: 1 validation error for black_scholesArguments sigma Field required [type=missing, input_value={'S': 100, 'K': 100, 'T':..., 'r': 0.05, 'vol': 0.2}, input_type=dict] For further information visit https://errors.pydantic.dev/2.14/v/missing`
- **0.5.0**: `{ "error": { "code": "RP402", "message": "black_scholes: unexpected argument(s) ['vol']; accepted: ['K', 'S', 'T', 'explain', 'kind', 'q', 'r', 'sigma']", "param": "vol", "hint": "Did you mean 'vol' → 'sigma'? `riskpy-tools schema black_scholes` lists them.", "docs": "https://slimboi34.github.io/Ris`

### `missing_argument`

- **0.4.0 (PyPI)**: `Error executing tool black_scholes: 1 validation error for black_scholesArguments sigma Field required [type=missing, input_value={'S': 100, 'K': 100, 'T': 1, 'r': 0.05}, input_type=dict] For further information visit https://errors.pydantic.dev/2.14/v/missing`
- **0.4.1 (main, unreleased)**: `Error executing tool black_scholes: 1 validation error for black_scholesArguments sigma Field required [type=missing, input_value={'S': 100, 'K': 100, 'T': 1, 'r': 0.05}, input_type=dict] For further information visit https://errors.pydantic.dev/2.14/v/missing`
- **0.5.0**: `{ "error": { "code": "RP403", "message": "black_scholes: missing required argument(s) ['sigma']", "param": "sigma", "hint": "Pass sigma (annual volatility as a decimal (0.2 means 20%)). Required: ['S', 'K', 'T', 'r', 'sigma'].", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp403" } }`

### `number_as_string`

- **0.4.0 (PyPI)**: `10.450583572185565`
- **0.4.1 (main, unreleased)**: `10.450583572185565`
- **0.5.0**: `{ "error": { "code": "RP404", "message": "S must be a number, got '100'", "param": "S", "hint": "Send S as a JSON number (100, not \"100\"); got a string.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp404" } }`

### `unknown_choice`

- **0.4.0 (PyPI)**: `Error executing tool basel_irb_capital`
- **0.4.1 (main, unreleased)**: `Error executing tool basel_irb_capital: asset_class must be one of corporate, sovereign, bank, retail_mortgage, retail_revolving, retail_other, got 'mortgage'`
- **0.5.0**: `{ "error": { "code": "RP108", "message": "asset_class must be one of corporate, sovereign, bank, retail_mortgage, retail_revolving, retail_other, got 'mortgage'", "param": "asset_class", "hint": "Use one of the values the message lists; they are case-insensitive where noted.", "docs": "https://slimb`

### `ragged_triangle`

- **0.4.0 (PyPI)**: `{ "method": "chain ladder (volume)", "total_latest": 330.0, "total_ultimate": 338.8235294117647, "total_reserve": 8.823529411764696, "factors": [ 1.5238095238095237, 1.0588235294117647 ], "origin": [ 0, 1 ], "latest": [ 150.0, 180.0 ], "ultimate": [ 158.8235294117647, 180.0 ], "reserve": [ 8.8235294`
- **0.4.1 (main, unreleased)**: `{ "method": "chain ladder (volume)", "total_latest": 330.0, "total_ultimate": 338.8235294117647, "total_reserve": 8.823529411764696, "factors": [ 1.5238095238095237, 1.0588235294117647 ], "origin": [ 0, 1 ], "latest": [ 150.0, 180.0 ], "ultimate": [ 158.8235294117647, 180.0 ], "reserve": [ 8.8235294`
- **0.5.0**: `{ "error": { "code": "RP502", "message": "triangle row 1 has 3 observed development periods but row 0 has 2; rows are origin periods, oldest first, so they cannot get longer going down", "param": "triangle", "hint": "Order the rows oldest origin first; pad the not-yet-observed cells of each row with`

### `age_outside_table`

- **0.4.0 (PyPI)**: `Error executing tool life_annuity_due`
- **0.4.1 (main, unreleased)**: `Error executing tool life_annuity_due: age 10 is outside the table, which runs from 20 to 130 (omega = 131)`
- **0.5.0**: `{ "error": { "code": "RP113", "message": "age 10 is outside the table, which runs from 20 to 130 (omega = 131)", "param": "x", "hint": "Ages run from 20 to 130 in this table.", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp113" } }`

### `missing_extra`

- **0.4.0 (PyPI)**: `Unknown tool: chain_ladder`
- **0.4.1 (main, unreleased)**: `Unknown tool: chain_ladder`
- **0.5.0**: `{ "error": { "code": "RP201", "message": "chain_ladder needs NumPy (the numpy package), which is not installed: pip install \"open-riskpy[sim]\"", "param": null, "hint": "pip install \"open-riskpy[sim]\"", "docs": "https://slimboi34.github.io/RiskPY/errors/#rp201" } }`

## How this was run

- 2026-10-15, macOS 26 (arm64), CPython 3.12.13, MCP SDK 2.2.0, NumPy 2.5.3, each target in its own
  virtualenv, every call from an empty working directory so the installed package is what runs.
- **0.4.0** is the wheel from PyPI (`pip install open-riskpy==0.4.0`).
- **0.4.1** was committed on `main` (7e322ea) but never tagged or published to PyPI; it is measured as its
  Python files over the 0.4.0 wheel. Its compiled-core fixes do not touch any of these cases. Its
  `riskpy.__version__` therefore reads 0.4.0, which is why the summary shows that string.
- **0.5.0** is this release's Python files over a compiled core built from the same sources.
- The command:

```bash
python benchmarks/diagnostics/run.py \
    --target "0.4.0 (PyPI)=/path/to/venv-0.4.0/bin/python" \
    --target "0.4.1 (main, unreleased)=/path/to/venv-0.4.1/bin/python" \
    --target "0.5.0=/path/to/venv-0.5.0/bin/python" \
    --markdown benchmarks/diagnostics/RESULTS.md
```

## Caveats

- The mistakes and the scoring rules were written by the same author as the 0.5.0 changes, so this
  measures what it measures: how the replies to *these* nineteen mistakes changed. The rules are
  mechanical (a regex for the code, an exact field or whole-word match for the parameter, a fixed
  phrase list or a non-generic `hint` field, a JSON parse) and are applied identically to every version.
- The first 0.5.0 run scored 71/76. It found two real gaps, both fixed before this run: a claims
  triangle whose newer row is longer than an older one was silently reserved (now refused, RP502),
  and the missing-NumPy error did not name the `numpy` package (it now does).
- Six of the nineteen mistakes (`vol_as_percent`, `rate_as_percent`, `coupon_as_percent`,
  `lgd_as_percent`, `number_as_string`, `ragged_triangle`) were not detected at all before 0.5.0:
  the tool returned a number (the table scores the MCP reply). `number_as_string` did fail on the
  CLI — in 0.4.0 with Python's own `'<=' not supported between instances of 'str' and 'int'`, in
  0.4.1 with "S must be a number" — but over MCP the SDK's validation converted `"100"` to `100.0`
  before the check ran and the agent got a price; 0.5.0 routes MCP calls through the same checks as
  every other path.
