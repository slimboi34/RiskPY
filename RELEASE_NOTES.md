# RiskPY v0.5.2 — Release Notes

**Release date:** 2026-10-17  
**Type:** Patch release — fixes found by a second round of testing (0.5.1 from PyPI)  
**Install:** `pip install -U open-riskpy`

---

## TL;DR

0.5.1 was installed from PyPI on Python 3.10, 3.12 and 3.13 and tested where the first round
did not look: the library API directly (not the tools), exception behaviour, thread and
process safety of the call trace, the MCP server under 1,200 pipelined requests, every
documented example and command, and the published wheels. Results are bit-identical to 0.4.0
on 268 sampled computations (the seeded C++ core included); the only numerical difference
from 0.5.0 is the parametric expected shortfall that 0.5.1 corrected, and that matches SciPy
to 2e-15. Three small bugs are fixed; no result changes.

---

## Fixed

- **A pickled `RiskPyError` lost its state.** Built-in exceptions pickle their instance dict;
  RiskPY's custom `__reduce__` rebuilt the error from its code, parameter and hint only, so
  `value`, notes added with `add_note()` and any attribute a caller set disappeared when an
  error crossed a process boundary (`multiprocessing`, `concurrent.futures`). They now travel
  with it; a `value` that cannot be pickled is dropped instead of making the error unpicklable.
- **`riskpy-tools trace` could end in a traceback.** A trace file holding bytes that are not
  UTF-8, a JSON line that is not an object, a non-numeric `ms`, or a `RISKPY_TRACE_FILE` that
  points at a directory crashed the command. It now skips what it cannot read.
- **`aggregate_loss` reported an overflow as `(34, 'Result too large')`** with no parameter
  (severity amounts near 1e300, or a coefficient of variation above 1e154). It is now RP304 on
  `severity_sd`, with a hint to express the amounts in larger units.
- Tests: the MCP stdio test now runs the server outside the checkout, so `python -m` imports
  the installed package rather than the source tree (it failed on any Python other than the
  one the in-tree extension was built for).

## Checked and left alone

- The OpenXLSX packaging fix holds: every 0.5.1 wheel on PyPI (26 files, macOS x86_64
  included) installs only `riskpy/` (plus `open_riskpy.libs/` on Windows), the extension links
  nothing but the system C/C++ runtime, and the sdist builds a working wheel.
- The ~280 converted raise sites keep the built-in type each one raised in 0.4.x (checked per
  function against 0.4.1's source and by calling 75 bad inputs on 0.4.0 and 0.5.1); messages
  are unchanged or extended, except that an `UnknownNameError` (a `KeyError`) prints without
  the quotes `KeyError` adds.
- Errors raised by the compiled core are still plain `ValueError`s (RP100 through the tools).
- The trace is exact under 8 threads × 6 processes × 500 calls, and the shared trace file
  gets every line intact. Entries from different processes carry the same `seq` numbers.

---

# RiskPY v0.5.1 — Release Notes

**Release date:** 2026-10-17  
**Type:** Patch release — fixes found by black-box testing 0.5.0 from PyPI  
**Install:** `pip install -U open-riskpy`

---

## TL;DR

0.5.0 was installed from PyPI into fresh environments and driven the way a new user and an AI
agent would: every CLI on a lean install, every one of the 30 tools over real MCP JSON-RPC with
valid arguments and with typical agent mistakes, and the eleven new calculators checked against
textbook formulas computed independently. The numbers were right with one exception, which this
release fixes; the rest are clearer errors, one more percentage guard, and packaging.

The only numerical change: `value_at_risk` with `method="parametric"` now reports the
**normal** expected shortfall. Every other result is unchanged.

---

## Fixed

- **`value_at_risk(method="parametric")` returned the historical expected shortfall** next to the
  normal VaR, so the two numbers came from different models (and the ES could be below the VaR).
  It now returns ES = −μ + σ·φ(z)/(1 − level), z = Φ⁻¹(1 − level), from the same sample mean and
  standard deviation as the VaR. The historical method is unchanged.
- **`bornhuetter_ferguson` accepted a loss ratio typed as a percentage.** `expected_loss_ratio=70`
  (meant: 70%) gave reserves a hundred times too big. The tool now refuses a loss ratio above 5
  (500%) with RP104 and the hint "Divide by 100 — 70% is 0.7"; 1.4 (a bad year) is still accepted.
  `tools.PLAUSIBLE` gains `"loss_ratio": 5.0`.
- **Errors named a parameter the tool does not have.** A library function names its own
  arguments; the agent only knows the tool's. Now: an empty triangle is `triangle` (was `data`),
  a non-whole maturity in `bootstrap_yield_curve` is `maturities` (was `maturity`), a negative
  `at` time is `at` (was `t`), a bad compounding is `from_compounding` or `to_compounding` (was
  `compounding`, so the agent could not tell which), and an unknown code in `riskpy_explain` is
  `topic` (was `code`).
- **A misleading hint for `aggregate_loss`.** Passing the lognormal's `severity_mu` /
  `severity_sigma` suggested `'severity_sigma' → 'severity_mean'?`. It now says the tool takes the
  claim size's mean and standard deviation on the money scale and gives the conversion from a
  LogNormal(mu, sigma).
- **The wheels installed OpenXLSX's headers, static libraries and CMake/pkg-config files** as
  top-level `include/` and `lib/` directories in `site-packages` (2.7 MB nobody needs, and
  generic names another package could collide with). The wheel now installs only the extension
  (`install.components` in pyproject.toml; checked with a local build).
- **0.5.0 shipped without macOS x86_64 wheels.** They built, but the upload job did not wait for
  the best-effort wheel jobs and started three minutes before they finished; Intel Macs fell back
  to compiling the sdist. The upload now waits for them (a failure there still does not block it).
- `riskpy-doctor`'s self-check listed `quant` as verified on a lean install, where it is skipped;
  it now names the modules it ran and the ones it skipped (and the JSON check carries `skipped`).
- `riskpy-verify` on a lean install showed each skip reason cut off at "…ships as an extra:". The
  missing-NumPy and missing-Matplotlib messages now put the quoted install line on their first
  line (`pip install "open-riskpy[sim]"` — unquoted, zsh treats the brackets as a glob).
- `riskpy-context` said a bond's `ytm` is an effective annual rate; it is compounded `frequency`
  times a year. `riskpy_explain life_annuity_due` said the table runs to 120; it runs to 130.

## Improved

- `convert_rate` (the tool and `riskpy.rates`) also accepts `"weekly"` (52), `"daily"` (365),
  `"annually"`, `"yearly"`, `"semi_annual"`, `"semiannually"`, `"semi-annually"` and
  `"continuously"`. Existing spellings mean what they did.

## Checked and left alone

Negative rates, a 0% rate, LGD of exactly 0 or 1, PD of 1, recovery of 0, a 120% volatility,
`T = 0`, rates up to the 100% cap, life ages up to 130 and terms of 0 all work as they should.
Two limits are deliberate and documented: rates above 100% (hyperinflation) and volatilities
above 500% are refused by the *tools* (the library functions take them), so `implied_vol` can
return a volatility above 5 that `black_scholes` the tool will then refuse.

---

# RiskPY v0.5.0 — Release Notes

**Release date:** 2026-10-15  
**Type:** Minor release — diagnostics, error codes, more tools  
**Install:** `pip install -U open-riskpy`

---

## TL;DR

RiskPY now says exactly what went wrong, to a person or to an agent. Every deliberate error
has a **stable code**, names the **parameter** and carries a **hint**; the tools, the CLI and the
MCP server return `{"error": {code, message, param, hint}}` instead of tracebacks, and refuse
percentages where decimals belong instead of computing a confident wrong answer.
`riskpy-doctor` checks an installation end to end, a **call trace** shows what an agent has
been doing, and **eleven more verified calculators** are tools (27 in all). Measured on a
scripted set of 19 agent mistakes (`benchmarks/diagnostics/`): **3/76 → 76/76** points for
code, parameter, hint and structured JSON (0.4.0 → 0.5.0).

**This release includes 0.4.1**, the security and hardening release committed on 2026-10-07
but never tagged or published to PyPI: if you are on 0.4.0, its fixes (below) reach you here.

No numerical result of any existing function changes. Exception types are unchanged — a bad
volatility is still a `ValueError` — and messages are unchanged apart from added detail.

---

## New

### Error codes (`riskpy.errors`, `riskpy-explain`)

- `RiskPyError` with `.code`, `.param`, `.hint`, `.to_dict()`; subclasses that are also the
  built-in type each site always raised: `InputError` (`ValueError`), `ArgumentTypeError`
  (`TypeError`), `UnknownNameError` (`KeyError`), `ConvergenceError` (`ValueError`),
  `MissingDependencyError` (`ImportError`). `tools.ToolUnavailable` is one too (RP201/RP202).
- Codes by family: **RP1xx** input domain, **RP2xx** missing dependency, **RP3xx** numerical,
  **RP4xx** tool call, **RP5xx** malformed data, **RP9xx** internal. 38 codes; the catalogue is
  in `riskpy/errors.py` and on the new [error codes](https://slimboi34.github.io/RiskPY/errors/)
  page. A test checks every code in the source is in the catalogue and on that page.
- About 280 raise sites across `quant`, `rates`, `credit`, `life`, `reserving`, `mc`, `viz` and
  the tools now carry a code and the parameter; the important ones carry a specific hint
  (`implied_vol` says what the no-arbitrage floor or ceiling is for the quote it was given).
- `riskpy-explain RP105`, `riskpy-explain --list [--json]`.

### Structured tool errors

- `tools.call(name, args, errors="return")` never raises for a failed call; it returns
  `{"error": {"code", "message", "param", "hint", "docs"}}`. The default `errors="raise"`
  raises as before.
- `riskpy-tools call` prints that object on stdout (exit 1) and a one-line summary with the
  hint on stderr. Invalid JSON arguments are RP406.
- **MCP calls now run through `tools.call`.** Errors come back as that JSON in a tool error.
  The SDK's own validation used to coerce `"100"` to `100.0` before RiskPY's checks ran; now a
  string where a number belongs is refused (RP404) on every path. The advertised input schemas
  are RiskPY's own, so every parameter has a description, and tools are annotated read-only.
- Unknown arguments get a suggestion (`vol` → `sigma`, `strike` → `K`), a misspelt tool a
  "did you mean", a missing argument its description.
- **Plausibility guards** (`tools.PLAUSIBLE`): the tools refuse a rate, yield, coupon,
  dividend yield or spread above 100% in absolute value (RP104), a volatility above 500% or
  below 0 (RP105) and an LGD outside [0, 1] (RP103). `r=5` used to price an option at a 500%
  rate. The library functions underneath still accept them.
- The triangle tools refuse a triangle whose newer origin has more observed periods than an
  older one (RP502); the chain ladder used to reserve it silently.

### `riskpy-doctor` and `riskpy.doctor()`

Checks the Python version; the installed distribution and whether the `riskpy` being imported
is that installation (a source checkout shadowing it is reported); that the compiled core
imports from the same place, computes a known answer, and is a 0.4.1+ build; NumPy, SciPy,
pandas, Matplotlib, the MCP SDK and Tkinter, each with the exact `pip` line; a 68-identity
subset of `riskpy-verify`; the tools; that the MCP server builds and where `riskpy-mcp` is; the
platform. Each check is `{id, status: ok|warn|fail, message, hint}`; `--json`; exit 1 on a fail.

### The call trace

- `tools.trace()` — the last 200 tool calls in the process: tool, abbreviated arguments,
  milliseconds, ok or error code and parameter, and whether it came from Python, the CLI or MCP.
  `tools.status()` adds version, tool availability, limits and counts by error code.
- `riskpy-tools trace [--json] [--limit N] [--file PATH]`; `RISKPY_TRACE_FILE` appends every
  call to a JSON-lines file so separate processes share one trace.
- Introspection tools, in the manifest and over MCP: `riskpy_status` (the above),
  `riskpy_doctor`, and `riskpy_explain`, which takes an error code or a tool name and returns
  the tool's formula and conventions.
- `black_scholes` and `bond_analytics` take `explain: true` and add the working: for
  Black–Scholes `d1`, `d2`, `N(d1)`, `N(d2)`, the discounted spot and strike, and the
  continuous-compounding convention. Without it the result is unchanged.

### Eleven new tools

Thin wrappers over functions the verification suite checks, each pinned in the tests to
numbers computed independently (closed forms, the SULT's Makeham law written out by hand, the
chain-ladder definition on the raw GenIns data, `statistics.NormalDist`):

`life_premium` (net premium by the equivalence principle, with the two APVs), `term_insurance`,
`endowment_insurance`, `temporary_annuity_due`, `convert_rate`, `bootstrap_yield_curve`,
`short_rate_zero_bond` (Vasicek / CIR), `hazard_from_spread`, `vasicek_loss_quantile`,
`value_at_risk` (historical or parametric, with expected shortfall; NumPy) and
`bornhuetter_ferguson` (NumPy).

### Measured: `benchmarks/diagnostics/`

`run.py` sends 19 realistic agent mistakes over MCP and the CLI to any installed version and
scores each reply for a code, the parameter, a hint and structured JSON:

| version | detected | code | param | hint | JSON | score |
|---|---|---|---|---|---|---|
| 0.4.0 (PyPI) | 13/19 | 0/19 | 3/19 | 0/19 | 0/19 | 3/76 (4%) |
| 0.4.1 (unpublished) | 13/19 | 0/19 | 10/19 | 3/19 | 0/19 | 13/76 (17%) |
| 0.5.0 | 19/19 | 19/19 | 19/19 | 19/19 | 19/19 | 76/76 (100%) |

Six mistakes — a volatility, rate, coupon or LGD given as a percentage, a number sent as a
string over MCP, an upside-down triangle — returned a number before 0.5.0. The suite was
written alongside these changes; `RESULTS.md` has the replies the agent actually saw and the
caveats.

## Changed

- `riskpy-context` adds a "When a call fails" section (the code families and what to do) and
  the new commands; the tool table counts calculators and introspection tools separately.
- `tools.manifest()` entries carry `kind` ("calculator" or "introspection").
- The `basel_irb_capital` tool's description listed asset classes `"mortgage"`, `"revolving"`
  and `"retail"`, which the function never accepted; it now lists the real ones
  (`retail_mortgage`, `retail_revolving`, `retail_other`).
- The CLIs replace characters the console cannot encode instead of crashing (Windows code pages).
- `riskpy-mcp --config` falls back to the `riskpy-mcp` beside the interpreter when it is not on
  `PATH`.
- `quant._check_option` refuses a non-string `kind` with RP108 instead of an `AttributeError`.
- Docs: new [Diagnostics](https://slimboi34.github.io/RiskPY/diagnostics/) and
  [Error codes](https://slimboi34.github.io/RiskPY/errors/) pages; "For AI agents" updated.

## From 0.4.1 (never published to PyPI)

0.4.1 was committed on 2026-10-07 and passed CI, but was never tagged, so it never reached
PyPI. Everything in it ships in 0.5.0; the full notes are in the 0.4.1 section below. In short:

- The agent tools check every argument against its type on every path (no `NaN`, `Infinity`,
  booleans or numeric strings where numbers belong) and bound the work one call can ask for
  (`tools.LIMITS`: trials, simulated claims, payment periods, triangle size, list and string
  lengths). A `NaN` frequency used to hang `aggregate_loss` forever.
- Over MCP, argument errors reach the agent instead of "Error executing tool".
- Compiled core: every `MonteCarloSimulator.simulate_*` refuses non-finite parameters and
  frequencies above 10⁹; `RiskEngine.attach_exporter` keeps the exporter alive (it used to
  read freed memory with a temporary exporter).
- CI and release: actions pinned to commit SHAs, least-privilege permissions, `zizmor` clean,
  grouped weekly Dependabot; `SECURITY.md`.

## Upgrading

Nothing to change for code that worked on 0.4.0. If you matched error messages, they are the
same text (some gained detail after the original). If you call the tools with percentages —
`r=5` for 5% — you now get RP104 instead of a number, which was wrong. If an MCP client sent
numbers as strings, send numbers.

---

# RiskPY v0.4.1 — Release Notes

**Release date:** 2026-10-07 — committed, never tagged or published to PyPI; ships in 0.5.0  
**Type:** Patch release — security and hardening  
**Install:** `pip install -U open-riskpy`

---

## TL;DR

A proactive audit of 0.4.0, which put RiskPY's calculators in front of AI agents.
An agent's arguments are untrusted input, and several could hang the process, exhaust
memory or reach undefined behaviour in the compiled core. The tools now check every
argument and bound the work one call can ask for; two crashes in the compiled core are
fixed; the release workflows are pinned to commit SHAs and run with least privilege.
No numerical result changes. No known exploitation; no CVE.

---

## Fixed

### Agent tools (`riskpy.tools`, `riskpy-tools`, `riskpy-mcp`)

- **A `NaN` frequency hung `aggregate_loss` forever.** Python's JSON parser accepts
  `NaN` and `Infinity`, and a Poisson draw with a `NaN` mean never returns. Every numeric
  argument of every tool must now be a finite number; booleans and numeric strings are
  refused rather than coerced, whole-number parameters must be whole, strings and lists
  have length limits.
- **Unbounded work.** `aggregate_loss` accepted any number of trials (memory) and any
  expected frequency (time); `bond_analytics` and `cds_par_spread` looped over
  `maturity × frequency` periods (`maturity = 1e300` never finished). Limits, in
  `riskpy.tools.LIMITS`: 1,000,000 trials and 10⁸ simulated claims per `aggregate_loss`
  call, 100,000 payment periods, 300 × 300 triangles. A call over a limit fails at once
  with a message that names it. The library functions underneath are unchanged and
  unbounded — the limits apply to the agent surface only.
- **The checks run on every path.** `@tool()` now registers a wrapper that checks the
  arguments against the type hints, so `tools.call`, the CLI, the MCP server and a
  direct `tools.black_scholes(...)` call behave the same. Schemas are unchanged.
- **Results are strict JSON.** `Infinity` in a result now becomes `null`, as `NaN`
  already did; `riskpy-tools call` refuses to print non-standard JSON.
- **Errors reach the agent.** Over MCP, the SDK hides the text of any exception that is
  not a `ToolError`, so an agent that sent a bad argument saw only "Error executing
  tool". Argument errors are now passed on as tool errors the agent can act on.
  `riskpy-tools` reports overflow as a one-line error instead of a traceback.
- `aggregate_loss` documented `seed` as making the answer reproducible; `seed = 0` (the
  default) draws a fresh seed each call. The description now says so; the default is
  unchanged. A negative seed is now refused with a clear message instead of a pybind11
  type error.

### Compiled core

- **`MonteCarloSimulator` accepted `NaN` parameters** (the checks were written `x < 0`,
  which `NaN` passes): a `NaN` expected frequency looped forever, other `NaN`s reached the
  standard-library distributions, whose behaviour is then undefined. Every `simulate_*`
  method now refuses non-finite parameters, and an expected frequency above 10⁹ (where the
  Poisson / negative-binomial count overflows an `int`) raises `ValueError`.
- **`RiskEngine.attach_exporter` kept a dangling pointer.** The engine stored a raw
  pointer to the exporter without keeping the Python object alive, so
  `engine.attach_exporter(ExcelExporter(...))` with a temporary freed the exporter while
  it was still attached, and a later `export_to_excel` read freed memory. The binding now
  ties the exporter's lifetime to the engine's (`py::keep_alive`).

## CI and release

- Every action pinned to a full commit SHA with the version in a trailing comment, at the
  current releases (checkout v7.0.1, setup-python v7.0.0, cache v6.1.0, upload-artifact
  v7.0.2, download-artifact v8.0.2, upload-pages-artifact v5.0.0, deploy-pages v5.0.1,
  cibuildwheel v4.3.0, pypi-publish v1.14.2).
- `persist-credentials: false` on every checkout; no job pushes with git.
- Docs: only the deploy job holds `pages: write` / `id-token: write` (the build, which runs
  the project's code, has `contents: read`); the unused `configure-pages` step is gone.
- Publish: no `${{ }}` expansions inside shell scripts; the GitHub Release job no longer
  checks out the repository while holding `contents: write`. File name, trigger,
  Trusted Publishing and the per-tag concurrency group are unchanged.
- CI: the ccache launcher expression meant to be empty on Windows never was
  (`cond && '' || x` always yields `x`); fixed.
- Nightly verification gets a concurrency group. `zizmor` (auditor persona) reports no
  findings.
- Dependabot: weekly, one grouped PR per ecosystem (GitHub Actions and pip), with a
  seven-day cooldown on new releases.
- `SECURITY.md`: how to report a vulnerability privately.

## Audit results

`bandit -r riskpy`: no issues. `pip-audit` over the runtime and test dependencies: no
known vulnerabilities. No `eval`, `exec`, `pickle`, YAML or `subprocess` in the library;
the only file writes are to paths the user names on the command line.

## Upgrading

`pip install -U open-riskpy`. Code that passed valid arguments sees no change. Code that
relied on a tool accepting `true` for a number, `"100"` for a number, or `NaN`, now gets
a `TypeError` / `ValueError`.

---

# RiskPY v0.4.0 — Release Notes

**Release date:** 2026-09-30  
**Type:** Minor release — the library for AI agents  
**Install:** `pip install -U open-riskpy` (`"open-riskpy[mcp]"` for the MCP server)  
**Documentation:** <https://slimboi34.github.io/RiskPY/agents/>

---

## TL;DR

An agent that can call functions needs a list of what it may call, a schema for
each call, and results it can pass on as data. 0.4.0 provides all three,
generated from the calculators' own signatures and docstrings, plus an MCP
server and a description of the library written for the model about to use
it. No numerical result changes; nothing new is required on a lean install.

---

## Added

### `riskpy.tools` and `riskpy-tools`

Sixteen tools, each a thin wrapper over a public function the verification
suite checks: `black_scholes`, `option_greeks`, `implied_vol`, `bond_analytics`,
`basel_irb_capital`, `expected_loss`, `merton`, `cds_par_spread`,
`annuity_certain`, `life_annuity_due`, `life_insurance`, `net_premium_reserve`,
`aggregate_loss`, `chain_ladder`, `mack_chain_ladder` and `verify`. Arguments are
plain numbers, strings and lists (rates as decimals, times in years); results
are JSON. `tools.manifest()` lists them with a JSON Schema per tool derived from
the type hints, with parameter descriptions from the docstrings; `tools.call(name,
arguments)` runs one. `aggregate_loss` takes a severity mean and standard
deviation on the money scale and converts to the lognormal parameters itself,
so the commonest severity mistake cannot be made through it. The two triangle
tools need NumPy and are reported as `available: false` with the `pip install`
line until it is present; calling one raises `ToolUnavailable` saying the same.

`riskpy-tools` lists them; `--json` prints the manifest; `call NAME '{...}'`
runs one; `schema NAME` prints its schema.

### `riskpy-mcp`

The tools over the Model Context Protocol on stdio, for Claude Code, Claude
Desktop, Cursor and any other client: `pip install "open-riskpy[mcp]"`, then
`riskpy-mcp`. `riskpy-mcp --config` prints the client configuration and the
`claude mcp add riskpy -- riskpy-mcp` line. Every available tool is exposed under
its own name with its schema and description.

### `riskpy-context`

A Markdown description of the library for the coding agent (or person) about
to use it: install lines per extra, the conventions that bite (rates are
decimals, `i` is effective annual while `r` is continuous, `LogNormal(mu,
sigma)` takes the underlying normal's parameters, seeds), the compiled core's
classes, every module's public functions with their signatures and first
docstring line, the tool table, and how to verify. Generated from the installed
package. `--json` gives it as data, `-o FILE` writes it.

### API

- New modules: `riskpy.tools`, `riskpy.context`, `riskpy.mcp_server`.
- New console scripts: `riskpy-tools`, `riskpy-context`, `riskpy-mcp`.
- New extra: `open-riskpy[mcp]` (the MCP SDK, `mcp>=2`).

---

# RiskPY v0.3.2 — Release Notes

**Release date:** 2026-09-29  
**Type:** Patch release — the verification suite on a lean install  
**Install:** `pip install -U open-riskpy`  
**Documentation:** <https://slimboi34.github.io/RiskPY/>

---

## TL;DR

`riskpy-verify` now passes on a bare `pip install open-riskpy`. A module whose
optional dependency is not installed — `mc`, `quant` and `reserving` without
NumPy, `viz` without Matplotlib — is reported as **skipped**, together with the
extra that would enable it, instead of being counted as a failed check. No
numerical result changes.

---

## Fixed

### A lean install reported three failures that were not failures

The core wheel deliberately has no Python dependencies, and the README says so.
But running `riskpy-verify` on exactly that install printed
`65/68 checks pass`, listed three `FAILED` entries reading
`import failed: riskpy.mc needs NumPy`, and exited 1 — which told a new user
the mathematics was wrong when only an extra was missing.

The suite now separates the two questions. `verify.collect()` returns the
checks it could compute *and* the modules it could not import; `Report.skipped`
carries them as `(module, reason)`; the text summary prints a `SKIPPED` section
with `pip install "open-riskpy[sim]"` (or `[viz]`) next to it; `--json` gains a
`skipped` list and `--markdown` a line above the table. `Report.ok` is
unchanged — no failed checks and at least one check — so a skipped module never
turns a red run green, and asking for a module that cannot run
(`riskpy-verify -m mc` without NumPy) still fails.

### API

- New: `riskpy.verify.collect(modules, oracle)` → `(checks, skipped)`.
- New: `Report.skipped: list[tuple[str, str]]`, and `"skipped"` in `to_dict()`.
- Unchanged: `verify.checks()`, `verify.run()`, `Report.ok`, the CLI flags and
  exit codes for runs that have failures.

---

# RiskPY v0.3.1 — Release Notes

**Release date:** 2026-09-15  
**Type:** Patch release — one pricing fix, a leaner sdist, internal cleanup  
**Install:** `pip install -U open-riskpy`  
**Documentation:** <https://slimboi34.github.io/RiskPY/>

---

## TL;DR

`ExposureRating.layer_premium` now prices ground-up layers; an attachment of 0
used to raise. The rest comes from the 2026-09-13 audit: duplicate helpers
merged, dead code removed, about twenty statements in the docs brought back in
line with the code, and an sdist that no longer ships the CI config. No result
changes except the fix below.

---

## Fixed

### `ExposureRating.layer_premium` raised at attachment 0

`layer_premium(p, attachment=0, limit)` raised `ValueError: Target limit must
be strictly positive`, although the method's own guard accepts
`attachment >= 0` and a ground-up layer is the commonest one to price. The
power-curve ILF is only defined for limits above zero, but the limited expected
value at zero is zero for any severity curve, so a layer from 0 now uses
ILF = 0:

```text
layer_premium(p, 0, L) = p · ILF(L) / ILF(base)
```

That is the limit as the attachment tends to 0, and layers stack:
[0, 100k] + [100k, 250k] = [0, 250k]. `increased_limits_factor(0)` itself
still raises.

### Smaller fixes

- The `UnderwritingApp` import error names what is actually missing (Tkinter,
  or Matplotlib/NumPy from the `[gui]` extra) instead of always blaming Tkinter.
- `viz.triangle` no longer calls Matplotlib's deprecated `Colormap.set_bad`.
- About twenty statements in the docs had drifted from the code. Among them:
  snippets that raised `NameError`, the lognormal μ, the Greeks and `summary()`
  output, the batch example's premium column, and the architecture pages.

---

## Changed

- The sdist leaves out `.github/` and `validate_riskpy.ipynb`. Neither is a
  build input; nothing else in the sdist changed.
- `rates` and `credit` import their identical validators from a new private
  module, `riskpy._validate`. `rates._check_finite` and the rest still resolve,
  and every message is unchanged.
- `LifeTable.from_force` integrates with `_special.simpson`. Its rates and
  errors are bitwise identical to 0.3.0 across 62 cases.
- The private `quant._normal_quantile` is gone. It had only wrapped
  `riskpy._special.norm_ppf` since 0.3.0, so call that instead.
- The Python < 3.8 `importlib_metadata` fallback is removed; the package
  requires 3.10.
- The `[dev]` extra adds pandas, so the `to_frame()` tests run in CI instead of
  skipping.
- CI: actions move to their current majors, every workflow gets least-privilege
  permissions and job timeouts, and Dependabot groups major action bumps into
  one PR.

---

## Testing

410 tests pass with none skipped, and all 125 verification checks pass.
A new regression test covers the ground-up layer: the closed form, stacking,
continuity at an attachment of 1e-9, the linear b = 0 curve, and that a
negative attachment still raises. The batch-app coverage the
root-level scripts used to give now lives in `tests/test_app_batch.py`.

---

## Upgrading

No public API changed. Code written against 0.3.0 runs unchanged; the one
different result is that ground-up layers return a premium instead of raising.

---

# RiskPY v0.3.0 — Release Notes

**Release date:** 2026-09-12  
**Type:** Feature release — four new fields of risk, a rewritten simulation engine, and a verification suite. One important fix.  
**Install:** `pip install -U open-riskpy`  
**Documentation:** <https://slimboi34.github.io/RiskPY/>

---

## TL;DR

RiskPY grows from a Monte Carlo engine with a quant module into a general risk
and actuarial library. Four new modules — **life contingencies**, **claims
reserving**, **interest rates** and **credit risk** — sit beside a rewritten
`riskpy.mc` whose twenty distributions now carry a full analytic layer, draw
Latin hypercube samples, and accept rank correlations. The chart library goes
from six charts to twenty-two. And every identity the library claims is now
checked by `riskpy.verify` — 125 checks against closed forms, published
tables and independent routes — on every push, nightly, and on the docs site.

The compiled core is untouched. `dependencies` is still empty. Nothing that
worked in 0.2.9 changed its result, with one exception below that was wrong
before and is right now.

```bash
pip install -U open-riskpy          # core, zero dependencies
pip install -U "open-riskpy[sim]"   # + NumPy
pip install -U "open-riskpy[viz]"   # + Matplotlib
```

---

## Fixed

### `quant.heston_price` returned the wrong price

The Fourier inversion evaluated the characteristic function of the log
*return* but used `log(K)` where it needed log-moneyness `log(K/S)`. At the
textbook parameters (`S = K = 100`) the price came back as roughly zero
instead of 8.93. The identity test that should have caught it — Heston
collapsing to Black–Scholes as vol-of-vol goes to zero — was in the suite and
was failing; it is now green, and the same identity is in the verification
suite so it cannot regress quietly.

### The examples did not run

`examples/catastrophe_model.py`, `examples/life_annuity_pricing.py` and three
documentation pages still called `MonteCarloSimulator.simulate_aggregate_loss(trials=…)`
as a static method, an API that changed in 0.2.4. They now construct the
simulator, and both examples run as part of checking this release.

### Smaller

- `Result.convergence()` started its running variance at one trial, where
  the sample variance is undefined, so the first band half-width was zero. It
  now starts at two and uses the unbiased estimator.
- `quant._normal_quantile` was Acklam's approximation on its own (relative
  error 1e-9). It now delegates to the new full-precision `norm_ppf`, which
  also tightened `parametric_var` and the reserving lognormal percentile.
- The light-mode chart palette failed the 3:1 contrast check on three
  colours. They are re-stepped; both palettes now pass all six categorical
  checks.

---

## Added

### `riskpy.life` — life contingencies

Pure Python, no NumPy. `LifeTable` from rates, from survivors, from Gompertz
or Makeham parameters, from an arbitrary force of mortality, or the AMLCR
Standard Ultimate Life Table. Every insurance and annuity present value —
whole life, term, pure endowment, endowment, deferred, increasing, mthly by
Woolhouse — with a `continuous=True` UDD adjustment. Net and gross premiums,
prospective reserves and reserve profiles, joint-life and last-survivor
statuses, commutation functions.

It reproduces the printed values in Dickson, Hardy & Waters to their printed
precision: `ä_40 = 18.4578`, `A_40 = 0.12106`, `l_100 = 6248.17`.

### `riskpy.reserving` — claims reserving

`Triangle` (ragged lists or NaN arrays, cumulative or incremental),
`chain_ladder` with volume, simple or regression averages and a tail,
`mack_chain_ladder` with process and parameter standard errors,
`bornhuetter_ferguson`, `cape_cod`, `expected_claims`, an England–Verrall ODP
`bootstrap_chain_ladder` whose result wraps as a `riskpy.mc.Result`, and
`fit_tail`. The GenIns triangle ships as `reserving.genins()`.

On GenIns it reproduces R's `ChainLadder` exactly: reserve 18,680,856, Mack
total standard error 2,447,095.

### `riskpy.rates` — interest rates

Discounting conventions and conversions; `YieldCurve` with linear or
log-linear interpolation, built directly, by bootstrap from par rates, or
parametrically (Nelson–Siegel, Svensson — which keep their formula, so the
limits hold exactly); `Bond` with price, yield, Macaulay and modified
duration, convexity, DV01, curve pricing and z-spread; Vasicek and CIR
closed-form bonds and curves, exact-transition Vasicek paths and
full-truncation CIR paths.

The CIR bond uses a `log1p` arrangement of the closed form that stays exact
as `σ → 0`; the textbook arrangement loses every digit below `σ ≈ 10⁻⁶`.

### `riskpy.credit` — credit risk

Expected and unexpected loss; `merton` (equity as a call on assets, distance
to default, spread); hazard rates, the credit triangle, CDS legs and par
spreads; the Vasicek / ASRF loss distribution, Basel asset correlations and
IRB capital (reproducing the BCBS table: 92.32% risk weight for PD 1%, LGD
45%); a one-factor Gaussian copula `credit_portfolio_loss` returning a
`Result`; `TransitionMatrix` with powers and cumulative default
probabilities, and a realistic S&P-style matrix to start from.

### `riskpy.mc` — rewritten

- **An analytic layer on every distribution**: `pdf`, `cdf`, `ppf`,
  `moments()` and `support()`, vectorised, scalar in → scalar out. Quantiles
  are accurate to around 1e-10 relative or better everywhere and to machine
  precision in the extreme tails — the upper tail inverts the survival
  function rather than the CDF, and heavy-shape betas and gammas whose lower
  quantiles sit near 1e-35 are seeded from the series expansion.
- **Six new distributions**: `Weibull`, `StudentT`, `Categorical`,
  `Mixture`, `Truncated`, and `Gamma.from_moments` /
  `LogNormal.from_median_cv` constructors.
- **Latin hypercube sampling**: `run(..., sampling="lhs")`. Same marginal,
  the noise on the mean cut by a factor of two to three.
- **Rank correlation**: `model.correlate("a", "b", 0.7)` or
  `simulate(..., correlation=...)`, imposed by Iman–Conover so the marginals
  are untouched. Jointly impossible correlations are rejected with the
  smallest eigenvalue named. `correlation_matrix()` and `iman_conover()` are
  public.
- **A copula hook**: `simulate(..., copula=obj)` for any object with `.dim`
  and `.uniforms(n, rng)`.
- **`Result` additions**: `median`, `skewness`, `kurtosis`, `cdf`,
  `percentiles`, `histogram`, `exceedance_curve`, `describe`,
  `sensitivity(method="pearson" | "contribution")`, `__len__`; quantile
  queries sort once and cache.
- **`Model` additions**: `correlate`, `sample`, `sweep` (re-run with one
  input swapped, same seed for every scenario).

### `riskpy.viz` — twenty-two charts

New: `density`, `cdf`, `qq`, `dashboard`, `spread`, `correlation`, `scatter`,
`triangle`, `development`, `reserve_range`, `survival`, `mortality`,
`reserve_profile`, `curve`, `allocation`, `waterfall`. Every chart is
factored so `dashboard` can draw into panels. Tick labels are compact
(`8.1m`, not `8,091,985`), `distribution` clips a heavy tail at the 99.9th
percentile and counts what it clipped, and direct labels de-collide with
leader lines. The palettes carry sequential and diverging ramps; both modes
pass the six categorical checks.

### `riskpy._special` — dependency-free special functions

Incomplete gamma and beta, the normal (with an `erfc` tail good to
`x ≈ −37`), Student-t and chi-square distributions, the Kolmogorov
distribution, digamma and trigamma, Brent, bisection, a bounded scalar
minimiser, Nelder–Mead and Simpson. Validated against SciPy to 1e-14 in the
tests without SciPy ever being a dependency. Vectorised variants iterate only
the unconverged points, which is what keeps a 200,000-point beta CDF at 37 ms.

### `riskpy.verify` — the verification suite

`verify.run()` returns a `Report`; `python -m riskpy.verify` and the
`riskpy-verify` console script print it, with `--bench` for timings and
`--json` / `--markdown` for machines. Each module contributes checks through
a `_verification_checks()` hook. It runs in CI on every push, in a new
nightly `verify.yml` workflow with the SciPy oracle and benchmarks, and the
docs workflow publishes the report.

---

## Changed

- `Poisson`, `NegativeBinomial`, `Bernoulli` and `Binomial` samples are
  float arrays (they were integer arrays), so every distribution returns the
  same dtype and formulas need no casts.
- `viz.distribution` clips its axis at the 99.9th percentile by default;
  pass `clip=None` for the full range.
- The light palette's series colours changed (contrast); the dark palette is
  as before.
- The `[dev]` extra now installs NumPy, Matplotlib and SciPy, since the test
  suite uses all three. `[docs]` requires mkdocs ≥ 1.6.
- New console script: `riskpy-verify`.
- Package summary and keywords describe the wider scope.

---

## Testing

407 test cases and 125 verification checks. The new tests follow the rule the
0.2.8 notes set out — identities before reference values — and add an
external witness where one exists: the special functions, the analytic
distribution layer and the truncated normal against SciPy; reserving against
R's `ChainLadder`; life against AMLCR; credit against the BCBS table.

---

## Upgrading

No API was removed. Code written against 0.2.9 runs unchanged, with three
things worth knowing: Heston prices are now correct, integer-distribution
samples are floats, and `viz.distribution` clips its axis unless told not to.

---

# RiskPY v0.2.9 — Release Notes

**Release date:** 2026-08-31  
**Type:** Maintenance re-release — no functional changes  
**Install:** `pip install -U open-riskpy`  
**Documentation:** <https://slimboi34.github.io/RiskPY/>

v0.2.9 is functionally identical to v0.2.8. There are no source changes; the
artifacts are simply rebuilt and republished with the current toolchain. There
is no need to upgrade, and nothing changes if you do.

---

# RiskPY v0.2.8 — Release Notes

**Release date:** 2026-07-26  
**Type:** Feature release — new modules, no breaking changes  
**Install:** `pip install -U open-riskpy`  
**Documentation:** <https://slimboi34.github.io/RiskPY/>

---

## TL;DR

Three new pure-Python modules on top of the C++ core, and a documentation site.
The compiled engine is untouched, `dependencies` is still empty, and nothing
that worked in 0.2.7 changed — the new layers ship as optional extras.

```bash
pip install -U open-riskpy          # core, zero dependencies
pip install -U "open-riskpy[sim]"   # + the Monte Carlo engine
pip install -U "open-riskpy[viz]"   # + the charts
```

---

## Added

### `riskpy.mc` — generic Monte Carlo

`MonteCarloSimulator` is fast but fixed: five specific actuarial scenarios, each
with its own signature. There was no way to say *"here is my formula, here are
my uncertain inputs, go"*. That is the common case, and it is now the front door.

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
```

The formula is called **once**, with arrays — 200k trials is one vectorised
expression, not a Python loop. `vectorised=False` falls back to looping for
formulas NumPy cannot express.

Fourteen distributions. Two carry alternative constructors because the raw
parameters are never the ones you have to hand:
`LogNormal.from_moments(mean, sd)` and
`NegativeBinomial.from_mean_dispersion(mean, dispersion)`.

`Result` answers what people actually ask of a run: `.var()`, `.tvar()`,
`.percentile()`, `.prob_above()`, `.standard_error`, `.convergence()`,
`.sensitivity()`, `.summary()`, `.to_frame()`.

### `riskpy.viz` — six charts, one theme

`distribution`, `exceedance`, `convergence`, `tornado`, `fan`, `compare`. Dark
by default, light available, both palettes validated for contrast and colour
vision deficiency. Each returns a Matplotlib `Figure` and none call `show()`, so
the same call works in a notebook, a script and a test.

### `riskpy.quant` — pricing, Greeks, paths, portfolio risk

Black–Scholes, Greeks and implied volatility are **pure Python** — no NumPy — so
they work in the lean install beside the C++ core. Greeks come back in the units
people quote them in: vega per percentage point, theta per calendar day.

Implied vol uses bisection rather than Newton, because vega collapses in the
wings and Newton diverges exactly where you most want an answer.

With NumPy: `gbm_paths` and `merton_jump_paths` for path simulation,
`historical_var`, `parametric_var` and `expected_shortfall` for portfolio risk.

`heston_price` prices European options by Fourier inversion of the
characteristic function — the concrete version of *"the formula is too big to
solve, so transform it instead of simulating it"*, and the same idea the
compiled `FourierTransform` already uses for aggregate loss. It uses the "little
Heston trap" formulation, which keeps the complex logarithm on its principal
branch; the textbook form is algebraically identical and numerically wrong past
a couple of years.

### Documentation site

<https://slimboi34.github.io/RiskPY/> — worked examples, every chart the library
produces, and the full API. The gallery is regenerated from the library on every
build, so the pictures cannot drift from the code, and a broken module fails the
docs build rather than publishing a stale screenshot.

---

## Changed

- `Documentation` in the package metadata now points at the site rather than a
  README anchor. This is the link PyPI renders.
- Package summary and keywords cover the quant side: Black–Scholes, Heston,
  value at risk.
- New `[sim]`, `[viz]` and `[docs]` extras. `[gui]` is unchanged.

---

## Testing

60 new cases across `tests/test_mc.py` and `tests/test_quant.py`, on top of the
existing suite. The pricing tests lean on identities rather than only on
published values, because identities catch classes of error a single reference
number does not:

- put–call parity, exactly, for both Black–Scholes and Heston
- every Greek checked as a numerical derivative of the price
- Heston collapsing to Black–Scholes as vol-of-vol → 0
- `parametric_var` against the known normal quantile
- Merton jump-diffusion keeping its drift, which is what the compensator is for

Statistical assertions use fixed seeds and generous tolerances: a test that
fails once a month teaches people to re-run it, and then they re-run it the day
it breaks for real.

---

## Upgrading

Nothing to do. No API changed, no dependency was added to the core install. The
new modules import lazily, so `import riskpy` on a machine with no NumPy behaves
exactly as it did in 0.2.7.

---

# RiskPY v0.2.7 — Release Notes

**Release date:** 2026-07-25  
**Type:** Packaging + release-pipeline release  
**Install:** `pip install -U open-riskpy`

---

## TL;DR

Correct package metadata, wheels for **Python 3.14** and **Linux aarch64**, and a
release pipeline that verifies what it ships before it ships it. No API changes —
upgrading is safe.

```bash
pip install -U open-riskpy
python -c "import riskpy; print(riskpy.__version__)"
```

---

## Fixed

- **Broken author metadata.** `pyproject.toml` shipped a literal placeholder
  (`[EMAIL_ADDRESS]`, no `@`) as the author email. v0.2.6 on PyPI carries it;
  0.2.7 has a real address.
- **`requires-python` claimed more than we shipped.** It said `>=3.8` while
  wheels covered only cp310–cp313, so pip on 3.8/3.9 silently fell through to a
  source build needing a full C++ toolchain. Now `>=3.10`, matching the wheels.
- **Stale pybind11 floor.** The CMake `FetchContent` fallback pinned v2.11.1,
  which predates Python 3.13 support, so a build without a pip-installed
  pybind11 could not target 3.13/3.14. Floor and fallback are now 2.12 / v3.0.4.
- **`CMAKE_CXX_COMPILER_LAUNCHER=ccache` was set on all CI platforms**, including
  the Windows image, which has no ccache.

## Added

- **cp314 wheels** — Python 3.14 is supported and classified.
- **linux/aarch64 wheels**, built on a native ARM runner rather than QEMU.
- **macos/x86_64 wheels** (cross-compiled) alongside arm64.
- **Windows CI** — the platform we ship wheels for is now actually tested.
- `make wheels` — reproduce the full CI wheel matrix locally.

## Changed

- **Release is one command.** `make release` tags and pushes; the tag push
  builds, publishes to PyPI, and cuts the GitHub Release with artifacts
  attached. Previously the tag push did nothing and a GitHub Release had to be
  created by hand — despite the workflow comments claiming otherwise.
- **New pre-publish gates**, so a bad release fails before upload, not after:
  - tag-vs-`pyproject` version check (fails in seconds, not after a 15-min matrix)
  - the sdist is installed and imported from a temp dir, proving it is self-contained
  - `twine check --strict` on every artifact
  - a warning if the version is already on PyPI
- **Best-effort extra architectures.** linux/aarch64 and macos/x86_64 cannot
  block a release; core linux/macos/windows wheels still must pass.
- **cibuildwheel 2.22 → 4.1.1.** Brings `manylinux_2_28`, automatic
  `delvewheel` DLL bundling on Windows, and `abi3audit` checks. The old
  `yum || apt` compiler shim is gone — the modern image already has C++17 and
  scikit-build-core injects cmake/ninja.
- **Wheel matrix moved into `pyproject.toml`** (`[tool.cibuildwheel]`) so local
  builds and CI cannot drift apart.
- **PEP 639 licensing** — `License-Expression: MIT` (metadata 2.4) replaces the
  deprecated `license = { text = ... }` table.
- **PyPI attestations** (PEP 740) are now published with each upload.
- Explicit `MACOSX_DEPLOYMENT_TARGET=11.0` for reproducible macOS wheel tags.

---

# RiskPY v0.2.6 — Release Notes

**Release date:** 2026-07-09  
**Type:** Feature + performance + packaging release  
**Install:** `pip install -U open-riskpy`

---

## TL;DR

Faster core math, headless-safe imports, new **FourierTransform** analytic aggregate-loss engine, stronger validation, expanded tests, and a push-button **GitHub Actions → PyPI** path (CI + wheels + Trusted Publishing).

```bash
pip install -U open-riskpy
python -c "import riskpy; from riskpy import FourierTransform; print(riskpy.__version__)"
```

---

## Added

### FourierTransform (C++)
- `fft` / `ifft` — iterative radix-2 Cooley–Tukey with zero-pad to power of 2  
- `convolve` — fast linear convolution  
- `compound_poisson_pmf` — aggregate loss PMF via $\hat{S}(t)=e^{\lambda(\hat{X}(t)-1)}$  
- Severity auto-normalized; non-finite inputs rejected; empty FFT returns empty  

### Packaging / DevX
- GitHub Actions **CI** (Ubuntu + macOS, Python 3.10–3.12)  
- **Publish** workflow: sdist + cibuildwheel wheels → PyPI Trusted Publishing  
- `Makefile` (`make install|test|smoke|build|release`)  
- `docs/PUBLISHING.md` rewritten for the one-command release path  

### Tests
- Full suites for Fourier, LossTriangle, RateAnalyzer, FactorModel extras, package headless import, RiskEngine  
- Suite size: **77** tests  

---

## Changed / Fixed

### Performance
- `RateAnalyzer.on_level_premiums`: O(n²) → **O(n)**  
- `LossTriangle.get_ultimate_losses`: O(years×periods) → **O(years)** with precomputed CDFs  
- `FactorModel.calculate`: field-keyed `unordered_map` lookup  
- `RiskEngine.get_fields`: return **const reference** (zero copy in C++)  

### Stability
- `import riskpy` no longer requires Tkinter (lazy `UnderwritingApp`)  
- Factor / rate / triangle / Fourier finite-value validation  
- OpenXLSX pinned to **v0.5.1**; shallow FetchContent; ccache-aware builds  
- Excel `create(..., XLForceOverwrite)` (non-deprecated API)  

---

## Prior releases

See git history and earlier sections of this file for v0.2.5 and below.  
*(v0.2.5 on PyPI was a packaging/CMake bump; v0.2.6 is the first full Fourier + perf ship.)*

---

# RiskPY v0.2.4 — Release Notes

**Release date:** 2026-05-01  
**Type:** Patch / bug-fix release  

### Fixed
- GUI Monte Carlo dispatch (`MonteCarloSimulator` instance method)  
- Headless smoke test  
- CSV batch newline handling  

```bash
pip install --upgrade open-riskpy
```
