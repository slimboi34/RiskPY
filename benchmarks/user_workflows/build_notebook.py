"""Build benchmarks/user_workflows/regression.ipynb from cells written here."""
import sys

import nbformat as nbf

OUT = sys.argv[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# RiskPY: 0.5.2 vs the development build, as a user would run it

This notebook runs RiskPY the way people actually use it — an options desk pricing a smile,
a risk team computing VaR and ES, a pensions actuary valuing annuities near zero rates, a
reserving actuary running Mack, a capital model, an AI agent calling the tools — and puts the
**published 0.5.2 wheel** next to the **development build** on every scenario.

* Every number is checked against a reference computed **independently of RiskPY** (SciPy,
  closed forms, published values: Mack 1993, AMLCR) — see `cases.py`.
* Each scenario runs in its own fresh interpreter, so a crash or hang in one cannot touch the next.
* *Everyday inputs* (no edge cases) are run through both versions to show nothing regressed.

**Run it** with the kernel from `~/RiskPY/.venv` (the development build installed there).
The 0.5.2 baseline is installed on first use into `~/.cache/riskpy-baseline/0.5.2`
(dependencies are shared with the kernel's environment).
""")

code("""
import math, os, sys, time, warnings
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

import cases
import riskpy

pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 200)
BASELINE = cases.baseline_path("0.5.2")
refs = cases.references()
print("development build:", os.path.dirname(riskpy.__file__))
print("baseline 0.5.2    :", BASELINE)
print(len(cases.CASES), "scenarios in", len({c.section for c in cases.CASES}), "areas")
""")

md("""
## 1. Every scenario, both versions

`✅` means the answer matches the independent reference to the stated tolerance, or the call
fails with the right structured error. `❌` says what happened instead.
""")

code("""
started = time.perf_counter()
old = cases.run_in(BASELINE)
new = cases.run_in(None)
print(f"ran {2 * len(cases.CASES)} scenario processes in {time.perf_counter() - started:.0f}s")


def shown(outcome):
    if "value" in outcome:
        v = outcome["value"]
        if isinstance(v, float):
            return f"{v:.10g}"
        text = repr(v)
        return text if len(text) < 40 else text[:37] + "..."
    if "timeout" in outcome:
        return f"hung > {outcome['timeout']:.0f}s"
    return f"{outcome['error']} {outcome.get('code') or ''}".strip()


def reference(case):
    e = case.expect
    if "raises" in e:
        return "error " + ("(C++ ValueError)" if e["raises"] == "core" else e["raises"] + "…")
    if "ref" in e:
        r = refs[e["ref"]] if isinstance(e["ref"], str) else e["ref"]
        return f"{r:.10g}" if isinstance(r, float) else (repr(r)[:37] + "...")
    return "finite result"


rows = []
for case, o, n in zip(cases.CASES, old, new):
    v_old, v_new = cases.judge(case, o, refs), cases.judge(case, n, refs)
    rows.append({
        "area": case.section, "scenario": case.label, "reference": reference(case),
        "0.5.2": shown(o), "dev": shown(n),
        "0.5.2 ok": "✅" if v_old == "pass" else "❌ " + v_old,
        "dev ok": "✅" if v_new == "pass" else "❌ " + v_new,
        "the 0.5.2 problem": case.why,
    })
results = pd.DataFrame(rows)

board = (results.assign(old_pass=results["0.5.2 ok"].eq("✅"), new_pass=results["dev ok"].eq("✅"))
         .groupby("area", sort=False)
         .agg(scenarios=("scenario", "size"), **{"0.5.2 passes": ("old_pass", "sum"), "dev passes": ("new_pass", "sum")}))
board.loc["TOTAL"] = board.sum()
board
""")

code("""
for area in results["area"].unique():
    display(Markdown(f"### {area}"))
    display(results[results["area"] == area].drop(columns=["area"]).set_index("scenario"))
""")

md("""
## 2. Regression check: everyday inputs

Ordinary, in-range calls — 60 options, 30 discount factors / IRB exposures / Merton firms /
CDS contracts, the SULT at 14 ages, the Taylor–Ashe triangle. Nothing here is an edge case,
so the two versions should agree to rounding. Where they differ on purpose, the table says why.
""")

code("""
every_new = cases.value_in(None, cases.EVERYDAY)
every_old = cases.value_in(BASELINE, cases.EVERYDAY)
expected_change = {
    "cds_par_spread": "exact accrual on default (0.5.2 used a midpoint rule); the dev values match a quadrature reference",
}
rows = []
for group, new_vals in every_new.items():
    old_vals = every_old[group]
    if group.startswith("implied_vol error"):
        rows.append({"group": group, "values": len(new_vals),
                     "max |0.5.2 − dev| (relative)": "—",
                     "0.5.2 worst |iv − σ|": f"{max(map(abs, old_vals)):.1e}",
                     "dev worst |iv − σ|": f"{max(map(abs, new_vals)):.1e}",
                     "verdict": "dev more accurate"})
        continue
    diff = max(abs(a - b) / max(abs(b), 1e-300) for a, b in zip(new_vals, old_vals))
    rows.append({"group": group, "values": len(new_vals), "max |0.5.2 − dev| (relative)": f"{diff:.1e}",
                 "0.5.2 worst |iv − σ|": "", "dev worst |iv − σ|": "",
                 "verdict": "identical to rounding" if diff < 1e-12 else expected_change.get(group, "CHANGED — investigate")})
everyday = pd.DataFrame(rows).set_index("group")
everyday
""")

md("""
## 3. Speed of the everyday calls

Microseconds per call (one process each, same machine, back to back). The edge-case fixes
added input checks and tighter tolerances; this is what they cost on ordinary inputs.
""")

code("""
t_new = cases.value_in(None, cases.TIMING)
t_old = cases.value_in(BASELINE, cases.TIMING)
timing = pd.DataFrame({"0.5.2 µs": t_old, "dev µs": t_new})
timing["dev / 0.5.2"] = timing["dev µs"] / timing["0.5.2 µs"]
timing.round(2)
""")

md("""
## 4. How a user runs it

### 4a. Options desk: a full volatility smile, wings included

Price a 20%-vol smile from 25% to 400% of spot with Black–Scholes, then recover the vols with
`implied_vol` — the round trip every vol desk runs. 0.5.2's absolute price tolerance gave up in
the cheap wings; its `erf`-based normal CDF priced the far puts at 0.
""")

code("""
strikes = [float(k) for k in np.geomspace(25, 400, 41)]
smile_code = f'''
from riskpy.quant import black_scholes, implied_vol
out = []
for K in {strikes!r}:
    kind = "put" if K < 100 else "call"
    p = black_scholes(100, K, 0.5, 0.02, 0.2, kind=kind)
    try:
        out.append([p, implied_vol(p, 100, K, 0.5, 0.02, kind=kind)])
    except Exception as exc:
        out.append([p, type(exc).__name__])
result = out
'''
smile_old, smile_new = cases.value_in(BASELINE, smile_code), cases.value_in(None, smile_code)


def vol_error(rows):
    return [abs(iv - 0.2) if isinstance(iv, float) else np.nan for _, iv in rows]


def price_error(rows, kinds):
    return [abs(p - cases.bs_ref(100, K, 0.5, 0.02, 0.2, kind=k)) / cases.bs_ref(100, K, 0.5, 0.02, 0.2, kind=k)
            for (p, _), K, k in zip(rows, strikes, kinds)]


kinds = ["put" if K < 100 else "call" for K in strikes]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for rows, label in ((smile_old, "0.5.2"), (smile_new, "dev")):
    axes[0].semilogy(strikes, np.maximum(price_error(rows, kinds), 1e-17), "o-", ms=3, label=label)
    axes[1].semilogy(strikes, np.maximum(vol_error(rows), 1e-17), "o-", ms=3, label=label)
axes[0].set(xscale="log", xlabel="strike (spot = 100)", ylabel="relative price error vs SciPy",
            title="Black–Scholes price")
axes[1].set(xscale="log", xlabel="strike (spot = 100)", ylabel="|recovered vol − 20%|",
            title="implied_vol round trip")
for ax in axes:
    ax.axhline(1e-8, color="grey", lw=0.6, ls=":")
    ax.set_xticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
    ax.minorticks_off()
    ax.legend()
plt.tight_layout()
plt.show()

print("strikes where implied_vol is more than 1e-6 off or fails:")
print("  0.5.2:", sum(not (isinstance(iv, float) and abs(iv - 0.2) < 1e-6) for _, iv in smile_old), "of", len(strikes))
print("  dev  :", sum(not (isinstance(iv, float) and abs(iv - 0.2) < 1e-6) for _, iv in smile_new), "of", len(strikes))
""")

md("""
### 4b. Credit: CDS spreads from investment grade to defaulting

Par spread for a flat hazard from 0.1% to 50 a year (R = 40%, r = 3%, 5y quarterly), against an
exact integral of protection and accrued premium (`cases.cds_ref`).
""")

code("""
hazards = [float(h) for h in np.geomspace(1e-3, 50, 30)]
cds_code = f'''
from riskpy.credit import cds_par_spread
result = [cds_par_spread(h, 0.4, 0.03, 5) for h in {hazards!r}]
'''
cds_old, cds_new = cases.value_in(BASELINE, cds_code), cases.value_in(None, cds_code)
exact = [cases.cds_ref(h, 0.4, 0.03, 5) for h in hazards]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].loglog(hazards, exact, "k-", lw=2, label="exact")
ax[0].loglog(hazards, cds_old, "o", ms=4, label="0.5.2")
ax[0].loglog(hazards, cds_new, "x", ms=5, label="dev")
ax[0].set(xlabel="hazard rate", ylabel="par spread", title="CDS par spread")
ax[1].loglog(hazards, [abs(a - e) / e for a, e in zip(cds_old, exact)], "o-", ms=3, label="0.5.2")
ax[1].loglog(hazards, [max(abs(a - e) / e, 1e-17) for a, e in zip(cds_new, exact)], "x-", ms=4, label="dev")
ax[1].set(xlabel="hazard rate", ylabel="relative error", title="error vs exact")
for a in ax:
    a.legend()
plt.tight_layout()
plt.show()
""")

md("""
### 4c. Capital model: a compound Poisson–lognormal book

Frequency Poisson(2), severity mean 1,000 and sd 2,000 on the money scale. The aggregate is
a **sum of N independent claims**, so its moments are E = λ·E[X] = 2,000 and
sd = √(λ·E[X²]) = √(2 × 5,000,000) = 3,162.

The formula `claim_count * severity` — the pattern in the README's first example — is **not**
that: it multiplies one claim size by the count, so every claim in a year is the same size and
the tail is overstated. The C++ `simulate_aggregate_loss` (and the `aggregate_loss` tool) draw
N claims; the FFT recursion on a discretised severity is an independent check.
""")

code("""
from riskpy.mc import simulate, Poisson, LogNormal, Bernoulli, Constant

sev = LogNormal.from_moments(1000, 2000)
mu, sigma = sev.mu, sev.sigma
theory_mean, theory_sd = 2 * 1000, math.sqrt(2 * (1000**2 + 2000**2))

naive = simulate(lambda n, s: n * s, 200_000, seed=42, n=Poisson(2.0), s=sev)
compound = np.asarray(riskpy.MonteCarloSimulator(200_000, 42).simulate_aggregate_loss(2.0, mu, sigma))

grid_step = 25.0
edges = np.arange(0, 400_000 + grid_step, grid_step)
cdf = np.array([sev.cdf(x + grid_step / 2) for x in edges])
pmf = np.diff(np.concatenate([[0.0], cdf]))
agg = np.asarray(riskpy.FourierTransform.compound_poisson_pmf(list(pmf), 2.0))
k = np.arange(agg.size) * grid_step
fft_mean = float((k * agg).sum())
fft_sd = math.sqrt(float((k * k * agg).sum()) - fft_mean**2)
fft_var995 = float(k[np.searchsorted(np.cumsum(agg), 0.995)])

compare = pd.DataFrame({
    "theory": [theory_mean, theory_sd, np.nan],
    "naive N × X (simulate)": [naive.mean, naive.std, naive.var(0.995)],
    "compound Σ X (C++ core)": [compound.mean(), compound.std(), float(np.quantile(compound, 0.995))],
    "FFT recursion": [fft_mean, fft_sd, fft_var995],
}, index=["mean", "sd", "VaR 99.5%"]).round(0)
display(compare)

# Point mass: a 1-in-100 total loss of 100. Expected shortfall at 95% is 100 × 0.01 / 0.05 = 20.
atom = simulate(lambda ev, sev: ev * sev, 200_000, seed=1, ev=Bernoulli(0.01), sev=Constant(100.0))
print(f"atom: VaR95 = {atom.var(0.95):.1f}, ES95 = {atom.tvar(0.95):.2f} (theory 20; 0.5.2 reported 1.0)")
""")

md("""
### 4d. Reserving: Mack on the Taylor–Ashe triangle, then a bootstrap
""")

code("""
from riskpy import reserving

tri = reserving.genins()
mack = reserving.mack_chain_ladder(tri)
table = pd.DataFrame({"reserve": np.round(mack.reserve, 0), "Mack s.e.": np.round(mack.se, 0)},
                     index=list(tri.origin) if hasattr(tri, "origin") else None)
display(table.T)
print(f"total reserve {float(np.sum(mack.reserve)):,.0f} (Mack 1993: 18,680,856); "
      f"total s.e. {mack.total_se:,.0f} (2,447,095)")

boot = reserving.bootstrap_chain_ladder(tri, n=2000, seed=7)
print(f"bootstrap: mean total {np.mean(boot.total):,.0f}, 99.5th pct {np.percentile(boot.total, 99.5):,.0f}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    flat = reserving.bootstrap_chain_ladder([[10, 20, 30, 40], [20, 40, 60], [30, 60], [40]], n=200, seed=1)
print("proportional triangle bootstrap (0.5.2: all NaN):", sorted({float(v) for v in np.round(flat.total, 6)})[:3])
""")

md("""
### 4e. An AI agent calling the tools

Agents send JSON: whole numbers arrive as floats, percentages turn up where decimals belong,
and an upstream NaN is a string away. Each call returns either strict JSON or a structured
error the agent can act on.
""")

code("""
from riskpy import tools

calls = [
    ("black_scholes", {"S": 100, "K": 105, "T": 0.5, "r": 0.03, "sigma": 0.2}),
    ("black_scholes", {"S": 100, "K": 105, "T": 0.5, "r": 0.03, "sigma": 20}),     # 20 for 20%
    ("aggregate_loss", {"expected_frequency": 2, "severity_mean": 1000, "severity_sd": 2000,
                        "trials": 50000.0, "seed": 3.0}),                           # JSON floats
    ("bootstrap_yield_curve", {"par_rates": [0.03, 0.035], "maturities": [1, 2], "frequency": 100000}),
    ("bond_analytics", {"face": 1e308, "coupon": 0.05, "maturity": 10, "ytm": 0.05}),
    ("convert_rate", {"rate": 0.05, "from_compounding": "monthly", "to_compounding": "continuous"}),
]
for name, args in calls:
    out = tools.call(name, args, errors="return")
    if isinstance(out, dict) and "error" in out:
        e = out["error"]
        print(f"{name:<22} → {e['code']} ({e.get('param')}): {e['message'][:90]}")
    else:
        text = repr(out if not isinstance(out, dict) else {k: out[k] for k in list(out)[:4]})
        print(f"{name:<22} → {text[:110]}")
""")

md("""
## 5. Verdict
""")

code("""
dev_fail = results[results["dev ok"] != "✅"]
old_fail = results[results["0.5.2 ok"] != "✅"]
regressions = results[(results["0.5.2 ok"] == "✅") & (results["dev ok"] != "✅")]
changed = everyday[everyday["verdict"].str.startswith("CHANGED")]
print(f"scenarios: {len(results)} | 0.5.2 fails {len(old_fail)} | dev fails {len(dev_fail)} | "
      f"regressions {len(regressions)}")
print(f"everyday groups that changed unexpectedly: {len(changed)}")
print(f"slowest relative to 0.5.2: {timing['dev / 0.5.2'].idxmax()} at {timing['dev / 0.5.2'].max():.1f}×")
assert dev_fail.empty and regressions.empty and changed.empty
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3 (RiskPY .venv)", "language": "python"}
nbf.write(nb, OUT)
print("wrote", OUT)
