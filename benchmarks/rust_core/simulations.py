"""The five ``MonteCarloSimulator`` methods: Rust core against closed forms and 0.8.0.

0.9.0 changed the random numbers, so seeded results cannot match 0.8.0. What must
match is the distribution. For each method, at several parameter points, this draws
200,000 trials from the core under test and from the C++ core of 0.8.0 (each in its own
interpreter) and reports:

* the sample mean and variance against the exact moments (z-scores);
* a Kolmogorov–Smirnov test against the exact distribution where it has one
  (lognormal paths, Pareto catastrophe sizes, binomial deaths via chi-square);
* a two-sample KS test of Rust against C++, and their 99% and 99.5% quantiles;
* that a seed repeats exactly and seed 0 does not; and extreme parameters, side by side.

    python simulations.py --cpp /venv-cpp/bin/python --rust /venv-rust/bin/python
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys

import numpy as np
from scipy import integrate, stats

TRIALS = 200_000

SNIPPET = """
import json, sys, numpy as np
import riskpy.cpp_underwriter as core
method, params, trials, seed = json.loads(sys.argv[1])
try:
    x = np.asarray(getattr(core.MonteCarloSimulator(trials, seed), method)(*params), dtype=float)
    np.save(sys.argv[2], x)
    print(json.dumps({"ok": True}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
"""


def run(python, method, params, trials=TRIALS, seed=7):
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    out = subprocess.run([python, "-I", "-c", SNIPPET, json.dumps([method, params, trials, seed]), path],
                         capture_output=True, text=True, check=True, cwd=tempfile.gettempdir()).stdout
    res = json.loads(out.strip().splitlines()[-1])
    if res["ok"]:
        res["x"] = np.load(path)
    os.remove(path)
    return res


def z(sample, mean, var):
    n = len(sample)
    zm = (sample.mean() - mean) / math.sqrt(var / n) if var > 0 else float(sample.mean() != mean) * math.inf
    mu4 = np.mean((sample - sample.mean()) ** 4)
    zv = (sample.var() - var) / math.sqrt(max(mu4 - var * var, 1e-300) / n) if var > 0 else 0.0
    return zm, zv


def life_moments(n, q, vol, benefit):
    """Moments of benefit × Binomial(n, p), p = min(1, q·max(0, S)), S ~ N(1, vol)."""
    if vol == 0:
        p = min(1.0, q)
        return n * p * benefit, n * p * (1 - p) * benefit ** 2
    pdf = stats.norm(1.0, vol).pdf
    lo, hi = 1.0 - 12 * vol, 1.0 + 12 * vol

    def p_of(s):
        return min(1.0, q * max(0.0, s))
    # integrate on [lo, hi] with the kinks at 0 and 1/q as break points
    pts = [b for b in (0.0, 1.0 / q if q > 0 else None) if b is not None and lo < b < hi]
    e_p = integrate.quad(lambda s: p_of(s) * pdf(s), lo, hi, points=pts, limit=200)[0]
    e_p2 = integrate.quad(lambda s: p_of(s) ** 2 * pdf(s), lo, hi, points=pts, limit=200)[0]
    mean = n * e_p
    var = n * (e_p - e_p2) + n * n * (e_p2 - e_p ** 2)   # E[Var | p] + Var[E | p]
    return mean * benefit, var * benefit ** 2


CASES = [
    # method, params, exact (mean, var) or None, exact distribution test or None
    ("simulate_aggregate_loss", [3.0, 0.0, 1e-9], (3.0, 3.0), "poisson_counts"),
    ("simulate_aggregate_loss", [5.0, 8.0, 0.5], None, None),
    ("simulate_aggregate_loss", [140.0, 9.3, 1.1], None, None),
    ("simulate_aggregate_loss", [0.2, 2.0, 2.0], None, None),
    ("simulate_life_portfolio", [1000, 0.01, 0.0, 1.0], None, "binomial"),
    ("simulate_life_portfolio", [50_000, 0.004, 0.25, 100_000.0], None, None),
    ("simulate_life_portfolio", [200, 0.6, 0.5, 1.0], None, None),
    ("simulate_economic_path", [100.0, 0.05, 0.2, 1], None, "lognormal"),
    ("simulate_economic_path", [50.0, -0.02, 0.35, 12], None, "lognormal"),
    ("simulate_health_claims", [10.0, 0.3, 2.0, 500.0], None, None),
    ("simulate_health_claims", [4.0, 2.5, 0.7, 1000.0], None, None),
    ("simulate_health_claims", [10.0, 1e-300, 1.0, 1.0], None, None),
    ("simulate_catastrophe_loss", [0.05, 1e6, 1.8], None, "pareto"),
    ("simulate_catastrophe_loss", [0.6, 10.0, 3.5], None, "pareto"),
]


def exact_moments(method, p):
    if method == "simulate_aggregate_loss":
        lam, mu, s = p
        return lam * math.exp(mu + s * s / 2), lam * math.exp(2 * mu + 2 * s * s)
    if method == "simulate_life_portfolio":
        return life_moments(*p)
    if method == "simulate_economic_path":
        p0, drift, vol, steps = p
        m = math.log(p0) + steps * (drift - vol * vol / 2)
        v = steps * vol * vol
        return math.exp(m + v / 2), (math.exp(v) - 1) * math.exp(2 * m + v)
    if method == "simulate_health_claims":
        lam, d, k, s = p
        ex, vx = k * s, k * s * s
        vn = lam + d * lam * lam
        return lam * ex, lam * vx + vn * ex * ex
    if method == "simulate_catastrophe_loss":
        q, xm, a = p
        if a <= 2:
            return q * a * xm / (a - 1) if a > 1 else math.inf, math.inf
        m1, m2 = a * xm / (a - 1), a * xm * xm / (a - 2)
        return q * m1, q * m2 - (q * m1) ** 2
    raise ValueError(method)


def exact_test(kind, method, p, x):
    if kind == "poisson_counts":
        counts = np.rint(x)
        lam = p[0]
        vals = np.arange(0, 20)
        e = stats.poisson(lam).pmf(vals) * len(x)
        e[-1] += stats.poisson(lam).sf(19) * len(x)
        o = np.array([(counts == v).sum() for v in vals[:-1]] + [(counts >= 19).sum()], dtype=float)
        keep = e > 5
        stat = (((o - e) ** 2) / e)[keep].sum()
        return f"chi2 vs Poisson({lam:g}) counts", stats.chi2.sf(stat, keep.sum() - 1)
    if kind == "binomial":
        n, q, _, b = p
        k = np.rint(x / b)
        dist = stats.binom(n, q)
        vals = np.arange(dist.ppf(1e-7), dist.ppf(1 - 1e-7) + 1)
        e = dist.pmf(vals) * len(x)
        o = np.array([(k == v).sum() for v in vals], dtype=float)
        keep = e > 5
        stat = (((o - e) ** 2) / e)[keep].sum()
        return "chi2 vs Binomial", stats.chi2.sf(stat, keep.sum() - 1)
    if kind == "lognormal":
        p0, drift, vol, steps = p
        ks = stats.kstest(np.log(x / p0), stats.norm(steps * (drift - vol * vol / 2), vol * math.sqrt(steps)).cdf)
        return "KS of log(path) vs normal", ks.pvalue
    if kind == "pareto":
        q, xm, a = p
        hits = x[x > 0]
        binom_p = stats.binomtest(len(hits), len(x), q).pvalue
        ks = stats.kstest(hits, stats.pareto(a, scale=xm).cdf)
        return f"KS sizes vs Pareto (event-rate test p={binom_p:.2g})", min(ks.pvalue, binom_p) if binom_p < 1e-4 else ks.pvalue
    raise ValueError(kind)


EXTREMES = [
    ("simulate_aggregate_loss", [1e7, 0.0, 1e-12], 3),
    ("simulate_aggregate_loss", [2.0, 700.0, 3.0], 1000),
    ("simulate_aggregate_loss", [1e-300, 0.0, 1.0], 1000),
    ("simulate_health_claims", [10.0, 1e300, 1.0, 1.0], 1000),
    ("simulate_health_claims", [1e9, 1e3, 1.0, 1.0], 1000),
    ("simulate_health_claims", [10.0, 1e-13, 1.0, 1.0], 1000),
    ("simulate_health_claims", [10.0, 0.5, 1e-300, 1.0], 1000),
    ("simulate_health_claims", [3.0, 0.5, 1e12, 1e-12], 1000),
    ("simulate_life_portfolio", [2_147_483_647, 0.5, 0.0, 1.0], 1000),
    ("simulate_life_portfolio", [1000, 0.0, 1e308, 1.0], 1000),
    ("simulate_life_portfolio", [1000, 1e-300, 1e308, 1.0], 1000),
    ("simulate_economic_path", [1.0, 0.0, 1e3, 1], 1000),
    ("simulate_economic_path", [1e300, 1.0, 0.0, 1000], 1000),
    ("simulate_catastrophe_loss", [1.0, 1.0, 1e-300], 1000),
    ("simulate_catastrophe_loss", [1.0, 1e300, 0.5], 1000),
]


def describe(res):
    if not res["ok"]:
        return res["error"]
    x = res["x"]
    finite = x[np.isfinite(x)]
    return (f"n={len(x)} zeros={int((x == 0).sum())} inf={int(np.isinf(x).sum())} nan={int(np.isnan(x).sum())} "
            f"mean(finite)={finite.mean() if len(finite) else float('nan'):.6g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--rust", required=True)
    ns = ap.parse_args()
    failures = 0
    print(f"{'method':<26} {'parameters':<30} {'z mean':>7} {'z var':>7}  {'exact test':<44} {'p':>8} "
          f"{'KS R vs C++ p':>13}  {'q99 R/C++':>10} {'q995 R/C++':>10}")
    for method, params, _, kind in CASES:
        r, c = run(ns.rust, method, params), run(ns.cpp, method, params)
        xr, xc = r["x"], c["x"]
        mean, var = exact_moments(method, params)
        zm, zv = z(xr, mean, var) if math.isfinite(var) else (z(xr, mean, 1.0)[0] * 0 + float("nan"), float("nan"))
        if not math.isfinite(var):
            zm, zv = float("nan"), float("nan")
        name, p = exact_test(kind, method, params, xr) if kind else ("", float("nan"))
        ks2 = stats.ks_2samp(xr, xc).pvalue
        q99 = np.quantile(xr, 0.99) / np.quantile(xc, 0.99) if np.quantile(xc, 0.99) else float("nan")
        q995 = np.quantile(xr, 0.995) / np.quantile(xc, 0.995) if np.quantile(xc, 0.995) else float("nan")
        bad = (abs(zm) > 5 if math.isfinite(zm) else False) or (abs(zv) > 5 if math.isfinite(zv) else False) \
            or (p < 1e-4 if math.isfinite(p) else False) or ks2 < 1e-4
        failures += bad
        print(f"{method:<26} {json.dumps(params):<30} {zm:7.2f} {zv:7.2f}  {name:<44} {p:8.3g} {ks2:13.3g}  "
              f"{q99:10.4f} {q995:10.4f}{'  FAIL' if bad else ''}")

    print("\nReproducibility (Rust):")
    for method, params, *_ in CASES[:: 3]:
        a, b = run(ns.rust, method, params, trials=10_000, seed=99), run(ns.rust, method, params, trials=10_000, seed=99)
        f1, f2 = run(ns.rust, method, params, trials=10_000, seed=0), run(ns.rust, method, params, trials=10_000, seed=0)
        same = np.array_equal(a["x"], b["x"])
        fresh = not np.array_equal(f1["x"], f2["x"]) or np.all(f1["x"] == 0)
        failures += (not same) + (not fresh)
        print(f"  {method:<26} seed 99 twice identical: {same};  seed 0 twice differs: {fresh}")

    print("\nExtreme parameters (Rust | C++ 0.8.0):")
    for method, params, trials in EXTREMES:
        r, c = run(ns.rust, method, params, trials=trials), run(ns.cpp, method, params, trials=trials)
        print(f"  {method} {json.dumps(params)} ×{trials}\n      Rust: {describe(r)}\n      C++:  {describe(c)}")
    print(f"\n{'all checks pass' if not failures else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
